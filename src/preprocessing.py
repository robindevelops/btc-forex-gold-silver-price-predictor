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
        self.processed_path = os.path.join(PROCESSED_DATA_DIR, self.config['filename'])
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
