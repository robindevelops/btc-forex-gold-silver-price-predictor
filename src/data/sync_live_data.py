import os
import sys
import pandas as pd
import yfinance as yf
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import PROCESSED_DATA_DIR, MODELS_DIR, ASSET_CONFIG
from src.data.preprocessing import DataCleaner
from src.data.external_data import fetch_all_external_data

def update_live_data(asset_name):
    """
    Fetches the latest data from Yahoo Finance, applies technical indicators,
    and scales it using the EXISTING scaler to prevent data leakage.
    """
    
    # Sync external data before preprocessing
    fetch_all_external_data()
    
    # 1. Fetch from yFinance
    try:
        config = ASSET_CONFIG.get(asset_name)
        ticker = yf.Ticker(config['ticker'])
        df = ticker.history(period="3y").reset_index()
        
        if df.empty:
            print(f"yFinance returned empty dataframe for {asset_name}. API might be down or rate-limited.")
            return False
            
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['timestamp', 'open', 'high', 'low', 'price', 'volume']
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    except Exception as e:
        print(f"API Error fetching {asset_name}: {str(e)}")
        return False
    
    # 2. Use DataCleaner to get indicators (SMA, EMA, RSI, MACD, BB)
    cleaner = DataCleaner(asset_name)
    cleaner.df = df
    
    # We suppress print statements from DataCleaner by redirecting stdout temporarily if we wanted, 
    # but printing is fine.
    cleaner.clean_data() 
    
    # 3. Save the updated raw features for Streamlit UI charts
    cleaner.df.to_csv(cleaner.processed_path)
    
    # 4. Scale the data using the EXISTING scaler (Do NOT refit!)
    prefix = 'btc' if asset_name == 'Bitcoin' else asset_name.lower()
    scaler_path = os.path.join(MODELS_DIR, f'{prefix}_scaler.pkl')
    
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        feature_cols = list(cleaner.df.columns)
        
        if len(feature_cols) != scaler.n_features_in_:
            print(f"Feature count mismatch for {asset_name}: scaler expects {scaler.n_features_in_}, got {len(feature_cols)}.")
            return False
            
        try:
            scaled_array = scaler.transform(cleaner.df[feature_cols])
            scaled_df = pd.DataFrame(scaled_array, columns=feature_cols, index=cleaner.df.index)
        except Exception as e:
            print(f"Error scaling data for {asset_name}: {e}")
            return False
        
        # Save as a specific live_scaled.csv file
        live_path = os.path.join(PROCESSED_DATA_DIR, f'{prefix}_live_scaled.csv')
        scaled_df.to_csv(live_path)
        return True
    return False

if __name__ == "__main__":
    for asset in ["Bitcoin", "Gold", "Silver"]:
        print(f"Syncing live data for {asset}...")
        update_live_data(asset)
        print("Done.\n")
