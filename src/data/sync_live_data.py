"""
Refresh the latest market data for live prediction.

Downloads go to data/raw_live/ (asset OHLCV, the external series and — for Silver — gold), and the
result is written to data/processed/<prefix>_live_features.csv, which inference prefers when present.
Nothing under data/raw/ or the frozen scaled splits is ever overwritten, so the reported results stay
reproducible while the dashboard and the daily job use up-to-date prices.

Every download already drops the running bar of the current day (complete-bar rule in
src/data/data_collection.py / external_data.py), so the live feature file always ends on a finished bar.
"""
import os
import sys
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import PROCESSED_DATA_DIR, RAW_LIVE_DIR, MODELS_DIR, ASSET_CONFIG, DATA_START_DATE, get_prefix
from src.data.preprocessing import DataCleaner
from src.data.data_collection import fetch_asset
from src.data.external_data import fetch_all_external_data

# external series each asset type's features are built from (keys of fetch_all_external_data's result; see DataCleaner.add_external_features)
REQUIRED_EXTERNAL = {'crypto': ('SP500', 'VIX', 'FearGreed'),
                     'commodity': ('DXY', 'CrudeOil', 'TreasuryYields', 'SP500', 'VIX')}


def missing_model_features(asset_name, columns):
    """Features the trained models expect (the train-fitted scaler's columns) that `columns` lacks; [] if no scaler exists yet."""
    path = os.path.join(MODELS_DIR, f'{get_prefix(asset_name)}_scaler.pkl')
    if not os.path.exists(path):
        return []
    expected = getattr(joblib.load(path), 'feature_names_in_', None)
    return sorted(set(expected) - set(columns)) if expected is not None else []


def update_live_data(asset_name, refresh_external=True):
    """
    Returns True on success. Safe to call from the dashboard; never raises on API failure.
    All-or-nothing: the live feature file is replaced only if every series this asset's features need was downloaded in THIS
    call (a failed download leaves the previous file in data/raw_live, and building today's row from it would silently repeat
    an old value — e.g. Silver's same-day gold return).
    """
    os.makedirs(RAW_LIVE_DIR, exist_ok=True)
    try:
        if refresh_external:
            ext = fetch_all_external_data(out_dir=RAW_LIVE_DIR) or {}
            failed = [s for s in REQUIRED_EXTERNAL[ASSET_CONFIG[asset_name]['type']] if ext.get(s) is None or ext[s].empty]
            if failed:
                print(f"External series {failed} were not downloaded; live data for {asset_name} not updated.")
                return False
        gold = None
        if asset_name == 'Silver':                       # silver's gold_return feature needs the same day's gold close
            gold = fetch_asset('Gold', DATA_START_DATE, out_dir=RAW_LIVE_DIR)
            if gold.empty:
                print("Gold (needed for Silver's gold_return) was not downloaded; live data for Silver not updated.")
                return False
        df = fetch_asset(asset_name, DATA_START_DATE, out_dir=RAW_LIVE_DIR)
        if df.empty:
            print(f"yFinance returned no data for {asset_name} (rate limit or outage).")
            return False
        if gold is not None and gold['timestamp'].max() < df['timestamp'].max():
            print("Gold's newest bar is older than Silver's; live data for Silver not updated.")
            return False
    except Exception as e:
        print(f"API error fetching {asset_name}: {e}")
        return False

    cleaner = DataCleaner(asset_name, raw_dir=RAW_LIVE_DIR)
    if not cleaner.load_data():
        return False
    try:
        cleaner.clean_data()
    except ValueError as e:                              # feature validation failed (e.g. a corrupt download)
        print(f"Live data for {asset_name} failed validation and was not written: {e}")
        return False
    missing = missing_model_features(asset_name, cleaner.df.columns)
    if missing:                                          # e.g. one external series failed to download on a fresh machine
        print(f"Live data for {asset_name} lacks model features {missing} and was not written "
              f"(the previous live file or the frozen dataset stays in use).")
        return False
    out = os.path.join(PROCESSED_DATA_DIR, f'{get_prefix(asset_name)}_live_features.csv')
    tmp = out + '.tmp'
    cleaner.df.to_csv(tmp)
    os.replace(tmp, out)                                 # atomic: a reader never sees a half-written file
    print(f"Live features for {asset_name} written to {out} (last complete bar {cleaner.df.index[-1].date()})")
    return True


if __name__ == "__main__":
    for asset in ASSET_CONFIG:
        update_live_data(asset, refresh_external=(asset == 'Bitcoin'))
