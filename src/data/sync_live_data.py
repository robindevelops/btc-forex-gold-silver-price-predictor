"""
Refresh the latest market data for live demonstration.

Downloads go to data/raw_live/ (asset OHLCV, the external series and — for Silver — gold), and the
result is written to data/processed/<prefix>_live_features.csv, which inference prefers when present.
Nothing under data/raw/ or the frozen scaled splits is ever overwritten, so the reported results stay
reproducible while the dashboard shows up-to-date prices.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import PROCESSED_DATA_DIR, RAW_LIVE_DIR, ASSET_CONFIG, DATA_START_DATE, get_prefix
from src.data.preprocessing import DataCleaner
from src.data.data_collection import fetch_asset
from src.data.external_data import fetch_all_external_data


def update_live_data(asset_name, refresh_external=True):
    """Returns True on success. Safe to call from the dashboard; never raises on API failure."""
    os.makedirs(RAW_LIVE_DIR, exist_ok=True)
    try:
        if refresh_external:
            fetch_all_external_data(out_dir=RAW_LIVE_DIR)
        if asset_name == 'Silver':                       # silver's gold_return feature needs fresh gold data
            fetch_asset('Gold', DATA_START_DATE, out_dir=RAW_LIVE_DIR)
        df = fetch_asset(asset_name, DATA_START_DATE, out_dir=RAW_LIVE_DIR)
        if df.empty:
            print(f"yFinance returned no data for {asset_name} (rate limit or outage).")
            return False
    except Exception as e:
        print(f"API error fetching {asset_name}: {e}")
        return False

    cleaner = DataCleaner(asset_name, raw_dir=RAW_LIVE_DIR)
    if not cleaner.load_data():
        return False
    cleaner.clean_data()
    out = os.path.join(PROCESSED_DATA_DIR, f'{get_prefix(asset_name)}_live_features.csv')
    cleaner.df.to_csv(out)
    print(f"Live features for {asset_name} written to {out} (last day {cleaner.df.index[-1].date()})")
    return True


if __name__ == "__main__":
    for asset in ASSET_CONFIG:
        update_live_data(asset, refresh_external=(asset == 'Bitcoin'))
