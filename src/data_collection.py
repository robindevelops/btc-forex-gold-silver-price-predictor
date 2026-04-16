import pandas as pd
import requests
import yfinance as yf
import os
from config import RAW_DATA_DIR, ASSET_CONFIG, DEFAULT_HISTORY_DAYS

def fetch_bitcoin_data(days=DEFAULT_HISTORY_DAYS):
    """
    Fetches historical Bitcoin data using yFinance (defaults to 3 years),
    bypassing the free CoinGecko 365-day API limitation.
    """
    return fetch_forex_data('Bitcoin')

def fetch_gold_data():
    """Fetches historical Gold data using yFinance."""
    return fetch_forex_data('Gold')

def fetch_silver_data():
    """Fetches historical Silver data using yFinance."""
    return fetch_forex_data('Silver')

def fetch_forex_data(asset_name):
    """
    Fetches historical forex/commodity data using yFinance.
    Standardizes output to ['timestamp', 'price'].
    """
    print(f"\nFetching data for {asset_name}...")
    config = ASSET_CONFIG.get(asset_name)
    if not config or config['source'] != 'yfinance':
        raise ValueError(f"Invalid asset {asset_name} for yFinance")
        
    ticker_symbol = config['ticker']
    filename = os.path.join(RAW_DATA_DIR, config['filename'])
    
    # Fetch data
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="3y") # 3 years to match default or max
    
    # Reset index to get Date as a column
    df = df.reset_index()
    
    # Clean and standardize
    df = df[['Date', 'Close', 'Volume']]
    df.columns = ['timestamp', 'price', 'volume']
    
    # Ensure datetime format (removing timezone if present for CSV compatibility)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    
    return df

def verify_data(df, asset_name):
    """Prints a brief check of the data as requested."""
    print(f"\n--- Verification: {asset_name} ---")
    print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price Range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"Average Volume: {df['volume'].mean():,.2f}")
    print(f"Null Values:\n{df.isnull().sum()}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    print("Starting data collection pipeline...")
    
    # 1. Bitcoin
    df_btc = fetch_bitcoin_data(days=DEFAULT_HISTORY_DAYS)
    verify_data(df_btc, "Bitcoin")
    
    # 2. Gold
    df_gold = fetch_gold_data()
    verify_data(df_gold, "Gold")
    
    # 3. Silver
    df_silver = fetch_silver_data()
    verify_data(df_silver, "Silver")
    
    print("\nWeek 1 Deliverable Check: 3 CSV files generated in data/raw/")
