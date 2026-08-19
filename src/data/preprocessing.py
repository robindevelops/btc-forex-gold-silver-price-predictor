import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, ASSET_CONFIG, BEST_LSTM_CONFIG

class DataCleaner:
    """
    Handles cleaning and basic preprocessing of raw timeseries data.
    """
    
    def __init__(self, asset_name):
        self.asset_name = asset_name
        self.config = ASSET_CONFIG.get(asset_name)
        if not self.config:
            raise ValueError(f"Asset {asset_name} not found in configuration.")
            
        self.raw_path = os.path.join(RAW_DATA_DIR, self.config['filename'])
        
        # Change output filename to _features.csv as requested
        if asset_name == 'Bitcoin':
            processed_filename = 'btc_features.csv'
        else:
            processed_filename = self.config['filename'].replace('_data', '_features')
            
        self.processed_path = os.path.join(PROCESSED_DATA_DIR, processed_filename)
        self.df = None

    def load_data(self):
        """Loads raw CSV and performs initial date conversion."""
        if not os.path.exists(self.raw_path):
            print(f"Error: Raw file not found for {self.asset_name} at {self.raw_path}")
            return False
            
        self.df = pd.read_csv(self.raw_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        print(f"Loaded {len(self.df)} rows for {self.asset_name}.")
        return True

    def add_moving_averages(self):
        """
        Calculates Exponential Moving Averages (EMA) for 14 and 30 days.
        Removed redundant SMAs and highly correlated timescales.
        """
        if self.df is None:
            return
            
        periods = [14, 30]
        for p in periods:
            # Exponential Moving Average
            self.df[f'EMA_{p}'] = self.df['price'].ewm(span=p, adjust=False).mean()
            
        print(f"Added EMA indicators for periods: {periods}")


    def add_rsi(self, window=14):
        """
        Manually calculates the Relative Strength Index (RSI).
        Uses Wilder's Smoothing method (exponential moving average).
        """
        if self.df is None:
            return

        delta = self.df['price'].diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)

        # Calculate using Wilder's smoothing method (alpha = 1 / window)
        ema_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
        ema_loss = loss.ewm(alpha=1/window, min_periods=window).mean()

        rs = ema_gain / ema_loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        print("Added RSI (14) indicator.")

    def add_macd(self, fast=12, slow=26, signal=9):
        """
        Manually calculates Moving Average Convergence Divergence (MACD).
        Standard periods: 12 (fast), 26 (slow), 9 (signal).
        """
        if self.df is None:
            return

        # MACD Line = Fast EMA - Slow EMA
        ema_fast = self.df['price'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['price'].ewm(span=slow, adjust=False).mean()
        self.df['MACD'] = ema_fast - ema_slow
        
        # Signal Line
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=signal, adjust=False).mean()
        print("Added MACD (12, 26, 9) indicator.")

    def add_advanced_ta(self):
        """
        Adds advanced technical indicators using the 'ta' library.
        Includes: Bollinger Bands, VWAP, ADX, ROC, and ATR.
        """
        if self.df is None:
            return
            
        import ta
        
        # Bollinger Bands
        indicator_bb = ta.volatility.BollingerBands(close=self.df['price'], window=20, window_dev=2)
        self.df['BB_Mid'] = indicator_bb.bollinger_mavg()
        self.df['BB_Upper'] = indicator_bb.bollinger_hband()
        self.df['BB_Lower'] = indicator_bb.bollinger_lband()
        
        # ATR (Average True Range)
        self.df['ATR'] = ta.volatility.average_true_range(
            high=self.df['high'], low=self.df['low'], close=self.df['price'], window=14
        )
        
        # VWAP (Volume Weighted Average Price) approximation daily
        self.df['VWAP'] = ta.volume.volume_weighted_average_price(
            high=self.df['high'], low=self.df['low'], close=self.df['price'], volume=self.df['volume'], window=14
        )
        
        # ADX
        adx = ta.trend.ADXIndicator(
            high=self.df['high'], low=self.df['low'], close=self.df['price'], window=14
        )
        self.df['ADX'] = adx.adx()
        
        # ROC
        self.df['ROC'] = ta.momentum.roc(close=self.df['price'], window=12)
        
        print("Added advanced TA features: BB, ATR, VWAP, ADX, ROC.")

    def add_lag_returns(self, lags=None):
        """
        Adds lagged log return features.
        
        Week 3 Feature: Captures short-term momentum at multiple timescales.
        Uses shift() to ensure each lag only references strictly past data.
        
        Args:
            lags: list of lag periods (default: [1, 2, 5, 10])
        """
        if self.df is None:
            return
        if lags is None:
            lags = [1, 2, 5, 10]
        
        for lag in lags:
            self.df[f'return_{lag}d'] = self.df['log_return'].shift(lag)
        
        print(f"Added lag return features for lags: {lags}")

    def add_rolling_volatility(self, windows=None):
        """
        Adds rolling standard deviation of log returns as volatility features.
        
        Week 3 Feature: Volatility clustering is one of the few genuinely robust
        patterns in financial data. Helps the model gauge current uncertainty.
        
        Args:
            windows: list of rolling window sizes (default: [10, 30])
        """
        if self.df is None:
            return
        if windows is None:
            windows = [10, 30]
        
        for w in windows:
            self.df[f'volatility_{w}d'] = self.df['log_return'].rolling(window=w).std()
        
        print(f"Added rolling volatility features for windows: {windows}")

    def add_calendar_features(self):
        """
        Adds cyclical day-of-week encoding using sin/cos transformation.
        
        Week 3 Feature: Markets show documented day-of-week effects (Monday/Friday).
        Cyclical encoding avoids artificial ordinal relationships (Mon=0, Tue=1, etc.)
        that a linear encoding would create.
        """
        if self.df is None:
            return
        
        day_of_week = self.df.index.dayofweek  # 0=Monday, 6=Sunday
        self.df['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        self.df['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        print("Added cyclical day-of-week features (dow_sin, dow_cos).")

    def add_volume_change(self):
        """
        Adds percentage change in volume vs. prior day.
        
        Week 3 Feature: Sudden volume spikes often precede or confirm
        real price moves vs. noise. Handles zero-volume edge cases.
        """
        if self.df is None:
            return
        
        # Use pct_change, then replace inf values (from division by zero volume) with 0
        self.df['volume_change'] = self.df['volume'].pct_change()
        self.df['volume_change'] = self.df['volume_change'].replace([np.inf, -np.inf], 0)
        
        print("Added volume_change feature.")

    def add_external_features(self):
        """
        Merges external macro/sentiment data into the feature set.
        
        Week 4 Feature: Addresses the audit finding that the model
        "captures zero macroeconomic factors" (Part 8B).
        
        Asset-specific mapping:
          - Gold/Silver: DXY return, Crude Oil return
          - Bitcoin: Fear & Greed Index
        """
        if self.df is None:
            return
        
        asset_type = ASSET_CONFIG[self.asset_name].get('type', '')
        
        # --- External Macro Features ---
        macro_sources = []
        if asset_type == 'commodity':
            macro_sources.extend([
                ('DXY', 'dxy_data.csv', 'dxy'),
                ('Crude Oil', 'crude_oil_data.csv', 'oil'),
                ('Treasury Yields', 'tnx_data.csv', 'tnx')
            ])
            
        # Global macro for all assets
        macro_sources.extend([
            ('S&P 500', 'sp500_data.csv', 'sp500'),
            ('VIX', 'vix_data.csv', 'vix')
        ])

        for ext_name, ext_file, col_prefix in macro_sources:
            ext_path = os.path.join(RAW_DATA_DIR, ext_file)
            if os.path.exists(ext_path):
                ext_df = pd.read_csv(ext_path, parse_dates=['timestamp'])
                ext_df = ext_df.set_index('timestamp')
                ext_df = ext_df.reindex(self.df.index).ffill()
                
                # Add log return of external source (not raw price level)
                self.df[f'{col_prefix}_return'] = np.log(
                    ext_df['price'] / ext_df['price'].shift(1)
                )
                self.df[f'{col_prefix}_return'] = self.df[f'{col_prefix}_return'].fillna(0)
                print(f"Added {ext_name} return feature ({col_prefix}_return).")
            else:
                print(f"WARNING: {ext_name} data not found at {ext_path}. Skipping.")
        
        # --- Fear & Greed Index for crypto (Bitcoin) ---
        if asset_type == 'crypto':
            fg_path = os.path.join(RAW_DATA_DIR, 'fear_greed_data.csv')
            if os.path.exists(fg_path):
                fg_df = pd.read_csv(fg_path, parse_dates=['timestamp'])
                fg_df = fg_df.set_index('timestamp')
                fg_df = fg_df.reindex(self.df.index).ffill()
                
                self.df['fear_greed'] = fg_df['fear_greed']
                self.df['fear_greed'] = self.df['fear_greed'].fillna(50)  # neutral default
                print("Added Fear & Greed Index feature.")
            else:
                print(f"WARNING: Fear & Greed data not found at {fg_path}. Skipping.")

    def validate_features(self):
        """
        Validates the final DataFrame to ensure no NaNs and correct shape.
        """
        if self.df is None:
            raise ValueError("No DataFrame to validate.")
            
        nan_count = self.df.isna().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Validation Error: Found {nan_count} NaN values in features.")
            
        if len(self.df) < 100:  
            raise ValueError(f"Validation Error: DataFrame shape too small: {self.df.shape}")
            
        print(f"Validation Passed: 0 NaNs found, shape {self.df.shape}.")

    def clean_data(self):
        """
        1. Removes duplicates
        2. Sorts by timestamp
        3. Sets timestamp as index
        4. Reindexes to daily frequency and forward fills
        5. Adds technical indicators
        """
        if self.df is None:
            return

        # 1. Remove duplicates and sort
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        duplicates_removed = initial_count - len(self.df)
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate timestamps.")

        # 2. Set index
        self.df = self.df.set_index('timestamp')

        # 3. Ensure Daily Frequency (reindex)
        full_range = pd.date_range(start=self.df.index.min(), end=self.df.index.max(), freq='D')
        self.df = self.df.reindex(full_range)
        
        # 4. Handle Missing Values (Forward Fill)
        for col in ['price', 'volume', 'open', 'high', 'low']:
            if col in self.df.columns:
                missing_count = self.df[col].isna().sum()
                if missing_count > 0:
                    self.df[col] = self.df[col].ffill()
                    print(f"Forward-filled {missing_count} missing {col} values for {self.asset_name}.")

        # 4.5 Add Log Returns
        # Log return = ln(P_t / P_{t-1})
        self.df['log_return'] = np.log(self.df['price'] / self.df['price'].shift(1))
        
        # 5. Add Technical Indicators
        self.add_moving_averages()
        self.add_rsi(window=14)
        self.add_macd(fast=12, slow=26, signal=9)
        self.add_advanced_ta()
        
        # 6. Week 3 Features: Lag returns, volatility, calendar, volume change
        self.add_lag_returns(lags=[1, 2, 5, 10])
        self.add_rolling_volatility(windows=[10, 30])
        self.add_calendar_features()
        self.add_volume_change()
        
        # 7. Week 4 Features: External macro/sentiment data
        self.add_external_features()
        
        # Drop NaNs created by rolling windows and lag features (warm-up period)
        self.df = self.df.dropna()

        # 6. Validate Features
        self.validate_features()

        # Optional: Name the index back to timestamp
        self.df.index.name = 'timestamp'
        print(f"Final cleaned count: {len(self.df)} rows.")

    def normalize_and_split(self, train_ratio=0.70, val_ratio=0.15):
        """
        Chronological split: 70% train, 15% val, 15% test.
        No random shuffle — time series must stay in order.
        Fits MinMaxScaler on training set ONLY to prevent data leakage.
        """
        if self.df is None:
            return

        n = len(self.df)
        feature_cols = list(self.df.columns)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = self.df.iloc[:train_end].copy()
        val_df = self.df.iloc[train_end:val_end].copy()
        test_df = self.df.iloc[val_end:].copy()

        # Print split sizes and date ranges for verification
        print(f"\n{'='*50}")
        print(f"Chronological Split for {self.asset_name}:")
        print(f"  Train: {len(train_df)} rows | {train_df.index.min().date()} → {train_df.index.max().date()}")
        print(f"  Val:   {len(val_df)} rows | {val_df.index.min().date()} → {val_df.index.max().date()}")
        print(f"  Test:  {len(test_df)} rows | {test_df.index.min().date()} → {test_df.index.max().date()}")
        print(f"{'='*50}\n")

        # Fit scaler on TRAINING data only — never on val/test
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(train_df[feature_cols])

        # Transform all three sets using the train-fitted scaler
        self.train_scaled = pd.DataFrame(
            self.scaler.transform(train_df[feature_cols]),
            columns=feature_cols, index=train_df.index
        )
        self.val_scaled = pd.DataFrame(
            self.scaler.transform(val_df[feature_cols]),
            columns=feature_cols, index=val_df.index
        )
        self.test_scaled = pd.DataFrame(
            self.scaler.transform(test_df[feature_cols]),
            columns=feature_cols, index=test_df.index
        )

        print("MinMaxScaler fitted on training data only (no data leakage).")

    def save_data(self):
        """
        Saves:
        1. Unscaled features CSV (for exploration)
        2. Scaled train / val / test CSVs
        3. Scaler object (.pkl) for inverse-transforming predictions
        """
        if self.df is None:
            return

        # 1. Save unscaled features
        self.df.to_csv(self.processed_path)
        print(f"Saved unscaled features to {self.processed_path}")

        # 2. Save scaled train / val / test
        base = self.processed_path.replace('_features.csv', '')
        train_path = f"{base}_train_scaled.csv"
        val_path = f"{base}_val_scaled.csv"
        test_path = f"{base}_test_scaled.csv"

        self.train_scaled.to_csv(train_path)
        self.val_scaled.to_csv(val_path)
        self.test_scaled.to_csv(test_path)
        print(f"Saved scaled train to {train_path}")
        print(f"Saved scaled val   to {val_path}")
        print(f"Saved scaled test  to {test_path}")

        # 3. Save scaler object for inverse-transform at prediction time
        scaler_filename = os.path.basename(base) + '_scaler.pkl'
        scaler_path = os.path.join(MODELS_DIR, scaler_filename)
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")

def create_sequences(data, seq_len=None, target_col='log_return'):
    """
    Sliding window sequence creator for LSTM input.

    Takes a scaled DataFrame and produces:
      X: (samples, seq_len, features) — the lookback window
      y: (samples, 1)                 — the next-day target price

    For each sample i, X[i] = rows[i : i+seq_len] (all features),
    and y[i] = the 'price' value at row[i+seq_len].
    """
    if seq_len is None:
        seq_len = BEST_LSTM_CONFIG['seq_len']
    target_idx = list(data.columns).index(target_col)
    values = data.values  # convert to numpy

    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i : i + seq_len])          # shape: (seq_len, features)
        y.append(values[i + seq_len, target_idx])   # next-day price

    return np.array(X), np.array(y).reshape(-1, 1)


def save_asset_sequences(asset_name, seq_len=None):
    """Generates and saves X, y numpy sequences to disk (.npy) for a given asset."""
    if seq_len is None:
        seq_len = BEST_LSTM_CONFIG['seq_len']
    prefix = ASSET_CONFIG[asset_name].get('filename').split('_')[0]
    if asset_name == 'Bitcoin':
        prefix = 'btc'

    train_path = os.path.join(PROCESSED_DATA_DIR, f"{prefix}_train_scaled.csv")
    val_path   = os.path.join(PROCESSED_DATA_DIR, f"{prefix}_val_scaled.csv")
    test_path  = os.path.join(PROCESSED_DATA_DIR, f"{prefix}_test_scaled.csv")

    train_df = pd.read_csv(train_path, index_col='timestamp', parse_dates=True)
    val_df   = pd.read_csv(val_path,   index_col='timestamp', parse_dates=True)
    test_df  = pd.read_csv(test_path,  index_col='timestamp', parse_dates=True)

    X_train, y_train = create_sequences(train_df, seq_len=seq_len)
    X_val,   y_val   = create_sequences(val_df,   seq_len=seq_len)
    X_test,  y_test  = create_sequences(test_df,  seq_len=seq_len)

    np.save(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_val.npy"), X_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_val.npy"), y_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_test.npy"), y_test)
    print(f"[{asset_name}] Saved .npy sequences to {PROCESSED_DATA_DIR}")

def load_dataset(asset_name):
    """
    Loader helper. 
    Loads the preprocessed X and y sequence arrays for the specified asset.
    Returns: X_train, y_train, X_val, y_val, X_test, y_test
    """
    prefix = ASSET_CONFIG[asset_name].get('filename').split('_')[0]
    if asset_name == 'Bitcoin':
        prefix = 'btc'
        
    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_train.npy"))
    X_val   = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_val.npy"))
    y_val   = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_val.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_test.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_test.npy"))
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_cleaning_pipeline():
    """Execution script for all configured assets."""
    print("Starting Data Cleaning Pipeline...\n" + "="*30)
    
    for asset in ASSET_CONFIG.keys():
        cleaner = DataCleaner(asset)
        if cleaner.load_data():
            cleaner.clean_data()
            cleaner.normalize_and_split(train_ratio=0.70, val_ratio=0.15)
            cleaner.save_data()
            # Generate and save LSTM sequences automatically
            save_asset_sequences(asset, seq_len=BEST_LSTM_CONFIG['seq_len'])
            print(f"Successfully processed {asset} entirely.\n" + "-"*30)


if __name__ == "__main__":
    run_cleaning_pipeline()
