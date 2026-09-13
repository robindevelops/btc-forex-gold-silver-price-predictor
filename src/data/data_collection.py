"""
Download daily OHLCV for Bitcoin (BTC-USD), Gold (GC=F) and Silver (SI=F) from Yahoo Finance.

    python src/data/data_collection.py                  # from config.DATA_START_DATE
    python src/data/data_collection.py --start 2015-01-01

Output: data/raw/<asset>_data.csv with columns timestamp, open, high, low, price (close), volume.
Gold/Silver are front-month futures: they trade on exchange days only (no weekend rows are
created — see preprocessing) and the yfinance volume column is contract-specific and noisy.
"""
import os
import argparse
import pandas as pd
import yfinance as yf

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import RAW_DATA_DIR, ASSET_CONFIG, DATA_START_DATE


def fetch_asset(asset_name, start=DATA_START_DATE, out_dir=RAW_DATA_DIR):
    """Download one asset's daily OHLCV into `out_dir` (data/raw by default; data/raw_live for demo refreshes)."""
    cfg = ASSET_CONFIG[asset_name]
    print(f"\nFetching {asset_name} ({cfg['ticker']}) from {start}...")
    try:
        df = yf.Ticker(cfg['ticker']).history(start=start)
    except Exception as e:
        print(f"Network error fetching data for {asset_name}: {e}")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'price', 'volume'])
    if df.empty:
        print(f"WARNING: empty response for {asset_name} (Yahoo rate limit?). Existing CSV left untouched.")
        return df
    df = df.reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = ['timestamp', 'open', 'high', 'low', 'price', 'volume']
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df = df.dropna(subset=['price'])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, cfg['filename'])
    df.to_csv(path, index=False)
    print(f"  {len(df)} rows, {df['timestamp'].min().date()} → {df['timestamp'].max().date()} saved to {path}")
    return df


# Backwards-compatible helpers
fetch_bitcoin_data = lambda **k: fetch_asset('Bitcoin')
fetch_gold_data = lambda: fetch_asset('Gold')
fetch_silver_data = lambda: fetch_asset('Silver')
fetch_forex_data = fetch_asset


def verify_data(df, asset_name):
    if df.empty:
        return
    print(f"--- {asset_name}: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}, "
          f"price ${df['price'].min():.2f}–${df['price'].max():.2f}, nulls={int(df.isnull().sum().sum())}, shape={df.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default=DATA_START_DATE)
    args = ap.parse_args()
    for asset in ASSET_CONFIG:
        verify_data(fetch_asset(asset, args.start), asset)
