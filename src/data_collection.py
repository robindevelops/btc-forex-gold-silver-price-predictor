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
    filename = os.path.join(RAW_DATA_DIR, ASSET_CONFIG['Bitcoin']['filename'])
    
    # YOUR CODE HERE
    pass

def fetch_forex_data(asset_name):
    """
    Fetches historical forex/commodity data using yFinance.
    
    TODO (Developer):
    1. Use the yf.Ticker(ticker_symbol) functionality to pull data.
    2. The ticker symbol for Gold is 'GC=F' and Silver is 'SI=F' (see config).
    3. Use the .history(period="max") or specific start/end dates.
    4. Clean the dataframe to only keep the 'Close' price and 'Date'.
    5. Rename columns as necessary to standardize with the BTC dataframe.
    6. Save to CSV in RAW_DATA_DIR.
    7. Return the DataFrame.
    """
    print(f"Fetching data for {asset_name}...")
    config = ASSET_CONFIG.get(asset_name)
    if not config or config['source'] != 'yfinance':
        raise ValueError(f"Invalid asset {asset_name} for yFinance")
        
    filename = os.path.join(RAW_DATA_DIR, config['filename'])
    ticker_symbol = config['ticker']
    
    # YOUR CODE HERE
    pass

if __name__ == "__main__":
    # Test script execution
    print("Starting data collection pipeline...")
    # df_btc = fetch_bitcoin_data()
    # df_gold = fetch_forex_data('Gold')
    # df_silver = fetch_forex_data('Silver')
    print("Done (uncomment function calls to run once implemented).")
