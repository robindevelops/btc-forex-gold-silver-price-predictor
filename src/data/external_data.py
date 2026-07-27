"""
External Data Collection Module.

Week 4: Fetches external macro and sentiment data to enrich
the prediction pipeline with signals the price chart alone cannot capture.

Sources:
  - US Dollar Index (DXY): Strongest known external driver of Gold/Silver
  - Crude Oil (CL=F): Commodity co-movement signal
  - Bitcoin Fear & Greed Index: Crypto-specific sentiment indicator
"""

import os
import sys
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import RAW_DATA_DIR, DEFAULT_HISTORY_DAYS

# External data configuration
EXTERNAL_SOURCES = {
    'DXY': {
        'ticker': 'DX-Y.NYB',
        'filename': 'dxy_data.csv',
        'description': 'US Dollar Index — primary driver of Gold/Silver pricing',
    },
    'CrudeOil': {
        'ticker': 'CL=F',
        'filename': 'crude_oil_data.csv',
        'description': 'WTI Crude Oil Futures — commodity co-movement signal',
    },
}


def fetch_yfinance_external(source_name):
    """
    Fetches external data from yfinance and saves to data/raw/.
    
    Args:
        source_name: Key in EXTERNAL_SOURCES dict (e.g. 'DXY', 'CrudeOil')
    
    Returns:
        DataFrame with columns ['timestamp', 'price']
    """
    config = EXTERNAL_SOURCES.get(source_name)
    if not config:
        raise ValueError(f"Unknown external source: {source_name}")
    
    print(f"\nFetching {source_name} ({config['description']})...")
    
    ticker = yf.Ticker(config['ticker'])
    df = ticker.history(period="3y")
    
    if df.empty:
        print(f"WARNING: No data returned for {source_name} ({config['ticker']})")
        return pd.DataFrame()
    
    df = df.reset_index()
    df = df[['Date', 'Close']]
    df.columns = ['timestamp', 'price']
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    
    filepath = os.path.join(RAW_DATA_DIR, config['filename'])
    df.to_csv(filepath, index=False)
    print(f"  Saved {len(df)} rows to {filepath}")
    print(f"  Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"  Price range: {df['price'].min():.2f} – {df['price'].max():.2f}")
    
    return df


def fetch_fear_greed_index():
    """
    Fetches the Bitcoin Fear & Greed Index from alternative.me (free, no API key).
    
    The index ranges from 0 (Extreme Fear) to 100 (Extreme Greed) and captures
    crypto-specific sentiment that price charts alone cannot provide.
    
    Returns:
        DataFrame with columns ['timestamp', 'fear_greed']
    """
    import urllib.request
    
    print("\nFetching Bitcoin Fear & Greed Index...")
    
    # Request ~1200 days of data (covers our 3-year window)
    url = "https://api.alternative.me/fng/?limit=1200&format=json"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode('utf-8')
        
        data = json.loads(raw)
        
        if 'data' not in data:
            print("WARNING: Fear & Greed API returned unexpected format")
            return pd.DataFrame()
        
        records = []
        for entry in data['data']:
            records.append({
                'timestamp': pd.to_datetime(int(entry['timestamp']), unit='s'),
                'fear_greed': int(entry['value']),
            })
        
        df = pd.DataFrame(records)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        filepath = os.path.join(RAW_DATA_DIR, 'fear_greed_data.csv')
        df.to_csv(filepath, index=False)
        print(f"  Saved {len(df)} rows to {filepath}")
        print(f"  Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
        print(f"  Value range: {df['fear_greed'].min()} – {df['fear_greed'].max()}")
        
        return df
        
    except Exception as e:
        print(f"WARNING: Failed to fetch Fear & Greed Index: {e}")
        print("  This is optional — BTC model will train without it.")
        return pd.DataFrame()


def fetch_all_external_data():
    """Fetches all external data sources and saves to data/raw/."""
    print("=" * 50)
    print("  FETCHING EXTERNAL DATA (Week 4)")
    print("=" * 50)
    
    results = {}
    
    # yfinance sources (DXY, Crude Oil)
    for source_name in EXTERNAL_SOURCES:
        results[source_name] = fetch_yfinance_external(source_name)
    
    # Fear & Greed Index
    results['FearGreed'] = fetch_fear_greed_index()
    
    print("\n" + "=" * 50)
    print("  External data collection complete.")
    for name, df in results.items():
        status = f"{len(df)} rows" if not df.empty else "FAILED"
        print(f"  {name}: {status}")
    print("=" * 50)
    
    return results


if __name__ == "__main__":
    fetch_all_external_data()
