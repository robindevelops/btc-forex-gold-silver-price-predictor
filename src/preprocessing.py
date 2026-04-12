import pandas as pd
import os
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, ASSET_CONFIG

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

    def save_data(self):
        """Saves the cleaned dataframe to the processed directory."""
        if self.df is not None:
            self.df.to_csv(self.processed_path)
            print(f"Saved processed data to {self.processed_path}")

def run_cleaning_pipeline():
    """Execution script for all configured assets."""
    print("Starting Data Cleaning Pipeline...\n" + "="*30)
    
    for asset in ASSET_CONFIG.keys():
        cleaner = DataCleaner(asset)
        if cleaner.load_data():
            cleaner.clean_data()
            cleaner.save_data()
            print(f"Successfully processed {asset}.\n" + "-"*30)

if __name__ == "__main__":
    run_cleaning_pipeline()
