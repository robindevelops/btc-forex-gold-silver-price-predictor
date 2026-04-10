import pandas as pd
import requests
import yfinance as yf
import os
from config import RAW_DATA_DIR, ASSET_CONFIG, DEFAULT_HISTORY_DAYS

def fetch_bitcoin_data(days=DEFAULT_HISTORY_DAYS):
    """
    Fetches historical Bitcoin data from CoinGecko API.
    
    API Endpoint: https://api.coingecko.com/api/v3/coins/bitcoin/market_chart
    Params: vs_currency=usd, days=days
    
    TODO (Developer): 
    1. Make the requests call to the endpoint.
    2. Extract the 'prices' array from the JSON response.
    3. Convert to a Pandas DataFrame with columns ['timestamp', 'price'].
    4. Convert 'timestamp' to datetime format.
    5. Save the DataFrame to CSV in RAW_DATA_DIR.
    6. Return the DataFrame.
    """
    print(f"Fetching Bitcoin data for the last {days} days...")
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Extract prices (format: [timestamp, price])
    prices = data['prices']
    
    # Create DataFrame
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    
    # Convert timestamp (ms) to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Save raw data to CSV
    filename = os.path.join(RAW_DATA_DIR, ASSET_CONFIG['Bitcoin']['filename'])
    df.to_csv(filename, index=False)
    
    print(f"Data saved to {filename}")
    return df

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
    df = df[['Date', 'Close']]
    df.columns = ['timestamp', 'price']
    
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
    print(f"Null Values:\n{df.isnull().sum()}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    print("Starting data collection pipeline...")
    
    # 1. Bitcoin
    df_btc = fetch_bitcoin_data(days=365)
    verify_data(df_btc, "Bitcoin")
    
    # 2. Gold
    df_gold = fetch_gold_data()
    verify_data(df_gold, "Gold")
    
    # 3. Silver
    df_silver = fetch_silver_data()
    verify_data(df_silver, "Silver")
    
    print("\nWeek 1 Deliverable Check: 3 CSV files generated in data/raw/")
