"""
Refresh the latest market data for live demonstration.

Writes data/processed/<prefix>_live_features.csv (used by inference when present).
It deliberately does NOT overwrite <prefix>_features.csv or the scaled splits, so the
frozen train/val/test data behind the reported results stays reproducible.
"""
import os
import sys
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import PROCESSED_DATA_DIR, ASSET_CONFIG, DATA_START_DATE, get_prefix
from src.data.preprocessing import DataCleaner
from src.data.external_data import fetch_all_external_data


def update_live_data(asset_name, refresh_external=True):
    """Returns True on success. Safe to call from the dashboard; never raises on API failure."""
    try:
        if refresh_external:
            fetch_all_external_data()
        cfg = ASSET_CONFIG[asset_name]
        df = yf.Ticker(cfg['ticker']).history(start=DATA_START_DATE).reset_index()
        if df.empty:
            print(f"yFinance returned no data for {asset_name} (rate limit or outage).")
            return False
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['timestamp', 'open', 'high', 'low', 'price', 'volume']
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    except Exception as e:
        print(f"API error fetching {asset_name}: {e}")
        return False

    cleaner = DataCleaner(asset_name)
    cleaner.df = df
    cleaner.clean_data()
    out = os.path.join(PROCESSED_DATA_DIR, f'{get_prefix(asset_name)}_live_features.csv')
    cleaner.df.to_csv(out)
    print(f"Live features for {asset_name} written to {out} (last day {cleaner.df.index[-1].date()})")
    return True


if __name__ == "__main__":
    for asset in ASSET_CONFIG:
        update_live_data(asset, refresh_external=(asset == 'Bitcoin'))
