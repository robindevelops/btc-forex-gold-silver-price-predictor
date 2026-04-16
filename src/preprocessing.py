import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, ASSET_CONFIG

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
        Calculates Simple Moving Averages (SMA) and Exponential Moving Averages (EMA)
        for standard periods: 7, 14, 30, 60 days.
        """
        if self.df is None:
            return
            
        periods = [7, 14, 30, 60]
        for p in periods:
            # Simple Moving Average
            self.df[f'SMA_{p}'] = self.df['price'].rolling(window=p).mean()
            # Exponential Moving Average
            self.df[f'EMA_{p}'] = self.df['price'].ewm(span=p, adjust=False).mean()
            
        print(f"Added SMA and EMA indicators for periods: {periods}")
        
        # Drop rows with NaNs created by rolling windows (e.g., the first 60 days)
        initial_len = len(self.df)
        self.df = self.df.dropna()
        print(f"Dropped {initial_len - len(self.df)} rows containing NaNs (warm-up period).")

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

    def add_bollinger_bands(self, period=20, std_dev=2):
        """
        Calculates Bollinger Bands.
        Middle Band = 20-day SMA
        Upper Band = 20-day SMA + (20-day StdDev * std_dev)
        Lower Band = 20-day SMA - (20-day StdDev * std_dev)
        """
        if self.df is None:
            return

        sma = self.df['price'].rolling(window=period).mean()
        std = self.df['price'].rolling(window=period).std()
        
        self.df['BB_Mid'] = sma
        self.df['BB_Upper'] = sma + (std * std_dev)
        self.df['BB_Lower'] = sma - (std * std_dev)
        print(f"Added Bollinger Bands (period={period}, std_dev={std_dev}).")

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
        for col in ['price', 'volume']:
            if col in self.df.columns:
                missing_count = self.df[col].isna().sum()
                if missing_count > 0:
                    self.df[col] = self.df[col].ffill()
                    print(f"Forward-filled {missing_count} missing {col} values for {self.asset_name}.")

        # 5. Add Technical Indicators
        self.add_moving_averages()
        self.add_rsi(window=14)
        self.add_macd(fast=12, slow=26, signal=9)
        self.add_bollinger_bands(period=20, std_dev=2)
        
        # Optional dropnas again as Bollinger and MACD adds more nans at the start
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

def create_sequences(data, seq_len=60, target_col='price'):
    """
    Sliding window sequence creator for LSTM input.

    Takes a scaled DataFrame and produces:
      X: (samples, seq_len, features) — the lookback window
      y: (samples, 1)                 — the next-day target price

    For each sample i, X[i] = rows[i : i+seq_len] (all features),
    and y[i] = the 'price' value at row[i+seq_len].
    """
    target_idx = list(data.columns).index(target_col)
    values = data.values  # convert to numpy

    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i : i + seq_len])          # shape: (seq_len, features)
        y.append(values[i + seq_len, target_idx])   # next-day price

    return np.array(X), np.array(y).reshape(-1, 1)


def save_asset_sequences(asset_name, seq_len=60):
    """Generates and saves X, y numpy sequences to disk (.npy) for a given asset."""
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
            save_asset_sequences(asset, seq_len=60)
            print(f"Successfully processed {asset} entirely.\n" + "-"*30)


if __name__ == "__main__":
    run_cleaning_pipeline()
