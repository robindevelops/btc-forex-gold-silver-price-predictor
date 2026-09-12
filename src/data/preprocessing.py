"""
Data cleaning, feature engineering, chronological split and scaling.

Design rules (see docs/METHODOLOGY.md):
  * Every feature at row t uses only information available at the close of day t.
  * The target is the NEXT row's log return (created in `create_sequences`), never a feature.
  * Assets keep their own trading calendar (no synthetic weekend rows for futures).
  * Only stationary features are model inputs; raw price levels are kept for charts.
  * The scaler is fitted on the training window only.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, ASSET_CONFIG, LEVEL_COLUMNS, TARGET_COL,
                    TRAIN_END, VAL_END, SEQ_LEN, TASKS, DEFAULT_TASK, get_prefix)


class DataCleaner:
    """Loads a raw OHLCV CSV, cleans it and adds features."""

    def __init__(self, asset_name):
        self.asset_name = asset_name
        self.config = ASSET_CONFIG.get(asset_name)
        if not self.config:
            raise ValueError(f"Asset {asset_name} not found in configuration.")
        self.prefix = get_prefix(asset_name)
        self.raw_path = os.path.join(RAW_DATA_DIR, self.config['filename'])
        self.processed_path = os.path.join(PROCESSED_DATA_DIR, f'{self.prefix}_features.csv')
        self.df = None
        self.scaler = None

    # ------------------------------------------------------------------ load
    def load_data(self):
        if not os.path.exists(self.raw_path):
            print(f"Error: Raw file not found for {self.asset_name} at {self.raw_path}")
            return False
        self.df = pd.read_csv(self.raw_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        print(f"Loaded {len(self.df)} rows for {self.asset_name}.")
        return True

    # ------------------------------------------------------- base indicators
    def add_moving_averages(self):
        for p in (14, 30):
            self.df[f'EMA_{p}'] = self.df['price'].ewm(span=p, adjust=False).mean()

    def add_rsi(self, window=14):
        """Wilder-smoothed RSI (0-100)."""
        delta = self.df['price'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        ema_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
        ema_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
        rs = ema_gain / ema_loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

    def add_macd(self, fast=12, slow=26, signal=9):
        ema_fast = self.df['price'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['price'].ewm(span=slow, adjust=False).mean()
        self.df['MACD'] = ema_fast - ema_slow
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=signal, adjust=False).mean()

    def add_advanced_ta(self):
        """Bollinger Bands, ATR, ADX, ROC via the `ta` library (all backward-looking)."""
        import ta
        bb = ta.volatility.BollingerBands(close=self.df['price'], window=20, window_dev=2)
        self.df['BB_Mid'] = bb.bollinger_mavg()
        self.df['BB_Upper'] = bb.bollinger_hband()
        self.df['BB_Lower'] = bb.bollinger_lband()
        self.df['ATR'] = ta.volatility.average_true_range(
            high=self.df['high'], low=self.df['low'], close=self.df['price'], window=14)
        self.df['ADX'] = ta.trend.ADXIndicator(
            high=self.df['high'], low=self.df['low'], close=self.df['price'], window=14).adx()
        self.df['ROC'] = ta.momentum.roc(close=self.df['price'], window=12)

    # -------------------------------------------------- stationary features
    def add_relative_features(self):
        """
        Convert price-level indicators into scale-free ratios so that the feature
        distribution does not drift when the price trends outside the training range.
        """
        p = self.df['price']
        self.df['ema14_ratio'] = p / self.df['EMA_14'] - 1
        self.df['ema30_ratio'] = p / self.df['EMA_30'] - 1
        self.df['macd_norm'] = self.df['MACD'] / p
        self.df['macd_hist_norm'] = (self.df['MACD'] - self.df['MACD_Signal']) / p
        width = (self.df['BB_Upper'] - self.df['BB_Lower'])
        self.df['bb_pctb'] = (p - self.df['BB_Lower']) / width.replace(0, np.nan)
        self.df['bb_width'] = width / self.df['BB_Mid']
        self.df['atr_norm'] = self.df['ATR'] / p
        self.df['hl_range'] = (self.df['high'] - self.df['low']) / p

    def add_lag_returns(self, lags=(1, 2, 5, 10)):
        """return_kd = log return observed k days ago (strictly past)."""
        for lag in lags:
            self.df[f'return_{lag}d'] = self.df['log_return'].shift(lag)
        # cumulative momentum over the last 20 days (monthly momentum / reversal literature)
        self.df['return_20d'] = self.df['log_return'].rolling(20).sum()

    def add_rolling_volatility(self, windows=(10, 30)):
        for w in windows:
            self.df[f'volatility_{w}d'] = self.df['log_return'].rolling(window=w).std()
        # HAR-style realised volatilities (Corsi 2009): daily, weekly, monthly RMS of returns, all backward-looking
        r2 = self.df['log_return'] ** 2
        self.df['rv_1d'] = np.sqrt(r2)
        self.df['rv_5d'] = np.sqrt(r2.rolling(5).mean())
        self.df['rv_22d'] = np.sqrt(r2.rolling(22).mean())
        # RiskMetrics EWMA volatility (lambda = 0.94), initialised on the first 22 days
        ewma_var = r2.ewm(alpha=1 - 0.94, min_periods=22, adjust=False).mean()
        self.df['ewma_vol'] = np.sqrt(ewma_var)

    def add_calendar_features(self):
        """Cyclical day-of-week. Only meaningful for the 7-day crypto market."""
        dow = self.df.index.dayofweek
        self.df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        self.df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    def add_volume_change(self):
        """Log change in volume, clipped. Only used for Bitcoin: yfinance futures volume
        for GC=F / SI=F is front-month contract volume and jumps at contract rolls."""
        v = self.df['volume'].replace(0, np.nan)
        self.df['log_volume_change'] = np.log(v / v.shift(1)).clip(-3, 3).fillna(0)

    def add_external_features(self):
        """
        Merge external macro/sentiment series as same-day log returns, aligned onto
        THIS asset's trading calendar with a forward fill (only past values are pulled).
        """
        asset_type = self.config['type']
        macro = []
        if asset_type == 'commodity':
            macro += [('dxy_data.csv', 'dxy'), ('crude_oil_data.csv', 'oil'), ('tnx_data.csv', 'tnx')]
        macro += [('sp500_data.csv', 'sp500'), ('vix_data.csv', 'vix')]

        for fname, col in macro:
            path = os.path.join(RAW_DATA_DIR, fname)
            if not os.path.exists(path):
                print(f"WARNING: {fname} not found, skipping {col}_return")
                continue
            ext = pd.read_csv(path, parse_dates=['timestamp']).set_index('timestamp')['price']
            ext = ext[~ext.index.duplicated()].sort_index()
            ext_ret = np.log(ext / ext.shift(1))               # computed on ITS OWN calendar
            self.df[f'{col}_return'] = ext_ret.reindex(self.df.index, method='ffill').fillna(0)

        if asset_type == 'crypto':
            path = os.path.join(RAW_DATA_DIR, 'fear_greed_data.csv')
            if os.path.exists(path):
                fg = pd.read_csv(path, parse_dates=['timestamp']).set_index('timestamp')['fear_greed']
                fg = fg[~fg.index.duplicated()].sort_index()
                self.df['fear_greed'] = fg.reindex(self.df.index, method='ffill').fillna(50)

        if self.asset_name == 'Silver':
            # Gold and silver settle at the same time; gold's same-day return is known at silver's close.
            path = os.path.join(RAW_DATA_DIR, 'gold_data.csv')
            if os.path.exists(path):
                g = pd.read_csv(path, parse_dates=['timestamp']).set_index('timestamp')['price']
                g = g[~g.index.duplicated()].sort_index()
                self.df['gold_return'] = np.log(g / g.shift(1)).reindex(self.df.index, method='ffill').fillna(0)

    # ------------------------------------------------------------ pipeline
    def clean_data(self):
        """1. de-duplicate & sort  2. calendar handling  3. log return  4. features  5. drop warm-up."""
        initial = len(self.df)
        self.df = self.df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        if initial - len(self.df):
            print(f"Removed {initial - len(self.df)} duplicate timestamps.")
        self.df = self.df.set_index('timestamp')
        self.df = self.df[self.df['price'] > 0]

        if self.config['type'] == 'crypto':
            # Crypto trades every day; a missing day is a data gap, fill from the previous day.
            full = pd.date_range(self.df.index.min(), self.df.index.max(), freq='D')
            missing = len(full) - len(self.df)
            self.df = self.df.reindex(full).ffill()
            if missing:
                print(f"Forward-filled {missing} missing calendar days for {self.asset_name}.")
        else:
            # Futures: keep exchange trading days only. NO synthetic weekend rows.
            print(f"{self.asset_name}: keeping {len(self.df)} exchange trading days (no calendar reindex).")

        self.df['log_return'] = np.log(self.df['price'] / self.df['price'].shift(1))

        self.add_moving_averages()
        self.add_rsi()
        self.add_macd()
        self.add_advanced_ta()
        self.add_relative_features()
        self.add_lag_returns()
        self.add_rolling_volatility()
        if self.config['type'] == 'crypto':
            self.add_calendar_features()
            self.add_volume_change()
        self.add_external_features()

        self.df = self.df.dropna()
        self.df.index.name = 'timestamp'
        self.validate_features()
        print(f"Final cleaned count: {len(self.df)} rows, {len(self.model_feature_columns())} model features.")

    def model_feature_columns(self):
        return [c for c in self.df.columns if c not in LEVEL_COLUMNS]

    def validate_features(self):
        nan_count = self.df.isna().sum().sum()
        if nan_count:
            raise ValueError(f"Validation Error: {nan_count} NaN values in features.")
        if len(self.df) < 100:
            raise ValueError(f"Validation Error: too few rows: {self.df.shape}")
        if np.isinf(self.df.select_dtypes(float).values).any():
            raise ValueError("Validation Error: inf values in features.")

    def normalize_and_split(self):
        """
        Chronological split by the frozen dates in config (no shuffling).
        The scaler is fitted on the training window ONLY and applied to val/test.
        Only model feature columns (stationary) are scaled and saved.
        """
        cols = self.model_feature_columns()
        train_df = self.df.loc[:TRAIN_END, cols]
        val_df = self.df.loc[pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1):VAL_END, cols]
        test_df = self.df.loc[pd.Timestamp(VAL_END) + pd.Timedelta(days=1):, cols]

        print(f"\n{'=' * 50}\nChronological Split for {self.asset_name}:")
        for name, d in (('Train', train_df), ('Val', val_df), ('Test', test_df)):
            print(f"  {name:5s}: {len(d):4d} rows | {d.index.min().date()} → {d.index.max().date()}")
        print('=' * 50)

        self.scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_df)
        self.train_scaled = pd.DataFrame(self.scaler.transform(train_df), columns=cols, index=train_df.index)
        self.val_scaled = pd.DataFrame(self.scaler.transform(val_df), columns=cols, index=val_df.index)
        self.test_scaled = pd.DataFrame(self.scaler.transform(test_df), columns=cols, index=test_df.index)
        print("MinMaxScaler fitted on training data only.")

    def save_data(self):
        self.df.to_csv(self.processed_path)
        base = os.path.join(PROCESSED_DATA_DIR, self.prefix)
        self.train_scaled.to_csv(f"{base}_train_scaled.csv")
        self.val_scaled.to_csv(f"{base}_val_scaled.csv")
        self.test_scaled.to_csv(f"{base}_test_scaled.csv")
        joblib.dump(self.scaler, os.path.join(MODELS_DIR, f'{self.prefix}_scaler.pkl'))
        print(f"Saved features, scaled splits and scaler for {self.asset_name}.")


# =========================================================================== sequences
def create_sequences(data, seq_len=SEQ_LEN, target_col=TARGET_COL):
    """
    Sliding windows for the recurrent models.
        X[i] = rows[i : i+seq_len]            (features up to and including day t)
        y[i] = rows[i+seq_len][target_col]    (log return of day t+1  →  ln(P_{t+1}/P_t))
    The window never contains row i+seq_len, so the target is never an input.
    """
    target_idx = list(data.columns).index(target_col)
    values = data.values
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i + seq_len])
        y.append(values[i + seq_len, target_idx])
    return np.array(X), np.array(y).reshape(-1, 1)


def _make_windows(values, seq_len):
    return np.stack([values[i:i + seq_len] for i in range(len(values) - seq_len + 1)])


def _task_target(log_return, task):
    """
    Real-space target series indexed by the date of day t (the last observed day).
        return_h : sum_{k=1..h} r_{t+k}
        vol_h    : ln sqrt( mean_{k=1..h} r_{t+k}^2 )
    Only FUTURE rows enter the target; features at t never see them.
    """
    h, kind = TASKS[task]['horizon'], TASKS[task]['kind']
    r = log_return
    if kind == 'return':
        fwd = sum(r.shift(-k) for k in range(1, h + 1))
    else:
        fwd = np.log(np.sqrt(sum(r.shift(-k) ** 2 for k in range(1, h + 1)) / h).clip(lower=1e-6))
    return fwd


def build_dataset(asset_name, seq_len=SEQ_LEN, task=DEFAULT_TASK, train_start=None):
    """
    The ONE loader used by tuning, training, stacking, evaluation and inference.

    Args:
        task        : key of config.TASKS ('return_1d' is served; others are experiments)
        train_start : optional date string — drop training rows before it (data-size experiments)

    Returns a dict with, for each split in ('train', 'val', 'test'):
        X_<split>      (n, seq_len, F)  scaled windows of features up to day t     (recurrent models)
        Xt_<split>     (n, F)           last row of each window = features at t   (tabular models)
        y_<split>      (n, 1)           standardised target (z-scored with TRAIN mean/std)
        y_real_<split> (n,)             target in real units (log return / log RV)
        dates_<split>  DatetimeIndex    day t (last observed day); the target covers t+1 … t+h
        target_dates_<split>             date of the last day the target covers (t+h)
        prev_<split>   ndarray          close on day t
        true_<split>   ndarray          close on day t+h (return tasks)
        naive_<split>  ndarray          task-specific naive forecast in real units (0 for returns,
                                        last realised vol for the vol task)
    plus 'columns', 'features', 'scaler', 'inv' (model → real), 'fwd' (real → model), 'task', 'horizon'.

    Splits are defined by the dates the target covers, so a training target never reaches past
    TRAIN_END and a validation target never reaches past VAL_END (exact purge/embargo for h > 1).
    """
    prefix = get_prefix(asset_name)
    read = lambda s: pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_{s}_scaled.csv'),
                                 index_col='timestamp', parse_dates=True)
    train_df, val_df, test_df = read('train'), read('val'), read('test')
    features = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_features.csv'),
                           index_col='timestamp', parse_dates=True)
    scaler = joblib.load(os.path.join(MODELS_DIR, f'{prefix}_scaler.pkl'))
    h = TASKS[task]['horizon']

    full = pd.concat([train_df, val_df, test_df])
    X = _make_windows(full.values, seq_len)              # window i ends at row i+seq_len-1
    dates_t = full.index[seq_len - 1:]                   # day t for each window
    price = features['price']
    target = _task_target(features['log_return'], task).reindex(dates_t)
    valid = target.notna().values
    X, dates_t, target = X[valid], dates_t[valid], target[valid]

    # Split by the dates the TARGET covers (t+1 … t+h): a training target must end on or before
    # TRAIN_END, a validation target must start after TRAIN_END and end on or before VAL_END, and a
    # test target must start after VAL_END. For h > 1 this is an exact purge/embargo by construction.
    pos = features.index.get_indexer(dates_t)
    first_target = features.index[pos + 1]
    last_target = features.index[np.minimum(pos + h, len(features) - 1)]
    t_end, v_end = pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END)
    masks = {'train': (last_target <= t_end) & ((dates_t >= pd.Timestamp(train_start)) if train_start else True),
             'val': (first_target > t_end) & (last_target <= v_end),
             'test': first_target > v_end}
    masks = {k: np.asarray(v, bool) for k, v in masks.items()}

    y_tr_real = target.values[masks['train']]
    mu, sd = float(y_tr_real.mean()), float(y_tr_real.std() or 1.0)
    fwd = lambda v: (np.ravel(v) - mu) / sd
    inv = lambda v: np.ravel(v) * sd + mu

    naive_feat = TASKS[task]['naive_feature']
    out = {'columns': list(full.columns), 'features': features, 'scaler': scaler, 'seq_len': seq_len,
           'task': task, 'horizon': h, 'kind': TASKS[task]['kind'], 'inv': inv, 'fwd': fwd, 'target_mean': mu, 'target_std': sd}
    for split, m in masks.items():
        d = dates_t[m]
        out[f'X_{split}'] = X[m]
        out[f'Xt_{split}'] = X[m][:, -1, :]
        out[f'y_real_{split}'] = target.values[m]
        out[f'y_{split}'] = fwd(target.values[m]).reshape(-1, 1)
        out[f'dates_{split}'] = d
        out[f'target_dates_{split}'] = last_target[m]
        out[f'prev_{split}'] = price.loc[d].values
        out[f'true_{split}'] = price.loc[last_target[m]].values if TASKS[task]['kind'] == 'return' else None
        if naive_feat:
            out[f'naive_{split}'] = np.log(features[naive_feat].loc[d].values.clip(1e-6))
        else:
            out[f'naive_{split}'] = np.zeros(len(d))
    return out


def unscale_target(y_scaled, scaler, target_idx):
    """Map a scaled log-return back to a real log-return (inverse of MinMaxScaler on one column)."""
    lo, hi = scaler.data_min_[target_idx], scaler.data_max_[target_idx]
    return np.ravel(y_scaled) * (hi - lo) + lo


def scale_target(y_real, scaler, target_idx):
    lo, hi = scaler.data_min_[target_idx], scaler.data_max_[target_idx]
    return (np.ravel(y_real) - lo) / (hi - lo)


def run_cleaning_pipeline():
    print("Starting Data Cleaning Pipeline...\n" + "=" * 30)
    for asset in ASSET_CONFIG:
        cleaner = DataCleaner(asset)
        if cleaner.load_data():
            cleaner.clean_data()
            cleaner.normalize_and_split()
            cleaner.save_data()
            print(f"Successfully processed {asset}.\n" + "-" * 30)


if __name__ == "__main__":
    run_cleaning_pipeline()
