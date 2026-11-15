"""
Implied-volatility series (free sources) and their close-time alignment.

    GVZ   CBOE Gold ETF Volatility Index — daily close at 16:15 ET (cdn.cboe.com daily history CSV, from 2009-09).
    DVOL  Deribit Bitcoin volatility index — daily candles from the public API (from 2021-03-24). The candle dated D
          closes at 00:00 UTC of D+1, the same instant as Yahoo's BTC-USD bar D.

    python src/volatility/implied_vol.py            # → data/raw_live/{gvz,dvol}_data.csv (live)
    python src/volatility/implied_vol.py --frozen   # → data/raw/, truncated to the frozen dataset's last day

Alignment (same rule as every macro series, config.EXTERNAL_SAME_DAY): GVZ is published after the 13:30 ET COMEX settlement,
so Gold and Silver use the PREVIOUS trading day's value; DVOL is known at the BTC close, so Bitcoin uses the same day's value.
The still-running bar of the current day is dropped at download time (complete-bar rule).
"""
import os
import sys
import io
import json
import time
import argparse
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import RAW_DATA_DIR, RAW_LIVE_DIR, ASSET_CONFIG, EXTERNAL_SAME_DAY
from src.data.market_calendar import drop_incomplete_bars

GVZ_URL = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/GVZ_History.csv'
DVOL_URL = ('https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC'
            '&start_timestamp={start}&end_timestamp={end}&resolution=1D')
DVOL_START = '2021-03-01'
IV_FILES = {'GVZ': 'gvz_data.csv', 'DVOL': 'dvol_data.csv'}
IV_FOR_ASSET = {'Bitcoin': 'DVOL', 'Gold': 'GVZ', 'Silver': 'GVZ'}     # no long free silver IV series: GVZ is the proxy
USER_AGENT = 'fyp-research-client'


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode('utf-8')


def fetch_gvz():
    df = pd.read_csv(io.StringIO(_get(GVZ_URL)))
    df.columns = [c.strip().lower() for c in df.columns]
    df = pd.DataFrame({'timestamp': pd.to_datetime(df['date'], format='%m/%d/%Y'), 'iv': pd.to_numeric(df['gvz'], errors='coerce')})
    df = df.dropna().drop_duplicates('timestamp').sort_values('timestamp')
    return drop_incomplete_bars(df, 'commodity')                  # US-session index: today's value final after the close


def fetch_dvol():
    rows, start = [], int(pd.Timestamp(DVOL_START).timestamp() * 1000)
    now, step = int(time.time() * 1000), 900 * 86_400_000           # the API returns at most 1,000 points per call
    while start < now:
        rows += json.loads(_get(DVOL_URL.format(start=start, end=min(start + step, now))))['result']['data']
        start += step
    df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close']).drop_duplicates('ts')
    df = pd.DataFrame({'timestamp': pd.to_datetime(df['ts'], unit='ms'), 'iv': df['close'].astype(float)}).sort_values('timestamp')
    return drop_incomplete_bars(df, 'crypto')                     # the candle dated today (UTC) is still running


def save(name, df, out_dir, until=None):
    if until is not None:
        df = df[df['timestamp'] <= pd.Timestamp(until)]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, IV_FILES[name])
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def fetch_all(out_dir=RAW_LIVE_DIR, until=None):
    """Download both series into `out_dir`. Returns {name: DataFrame or None}; never raises on a network failure."""
    out = {}
    for name, fn in (('GVZ', fetch_gvz), ('DVOL', fetch_dvol)):
        try:
            df = fn()
            if df.empty:
                raise ValueError('empty response')
            save(name, df, out_dir, until)
            out[name] = df
        except Exception as e:                                    # network / format problem: keep the previous file
            print(f"WARNING: {name} implied volatility not downloaded ({e}); the previous file stays in use.")
            out[name] = None
    return out


def read_iv(asset, raw_dir=RAW_DATA_DIR):
    """The raw implied-volatility series for `asset` (index = its own date), or None if the file is missing."""
    path = os.path.join(raw_dir, IV_FILES[IV_FOR_ASSET[asset]])
    if not os.path.exists(path):
        return None
    s = pd.read_csv(path, parse_dates=['timestamp']).set_index('timestamp')['iv']
    return s[~s.index.duplicated()].sort_index()


def align_iv(asset, index, raw_dir=RAW_DATA_DIR):
    """
    Implied volatility known at the close of each day in `index` (the asset's own calendar):
      Bitcoin        → DVOL of the same day (exact date match; NaN where the series has no value — no forward fill)
      Gold / Silver  → GVZ of the latest date STRICTLY before day t (published after the settlement)
    NaN before the series starts. Values are annualised percentages, as published.
    """
    s = read_iv(asset, raw_dir)
    index = pd.DatetimeIndex(index)
    if s is None:
        return pd.Series(np.nan, index=index)
    if EXTERNAL_SAME_DAY[ASSET_CONFIG[asset]['type']]:
        return s.reindex(index)
    pos = s.index.searchsorted(index, side='left') - 1
    vals = np.where(pos >= 0, s.values[np.clip(pos, 0, None)], np.nan)
    # a value older than 5 calendar days (long gap / series ended) is treated as missing rather than carried forward
    age = (index - s.index[np.clip(pos, 0, None)]).days
    vals = np.where((pos >= 0) & (age <= 5), vals, np.nan)
    return pd.Series(vals, index=index)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--frozen', action='store_true', help='write data/raw/, truncated to the frozen dataset (results reproducibility)')
    args = ap.parse_args()
    if args.frozen:
        until = {n: None for n in IV_FILES}
        btc_end = pd.read_csv(os.path.join(RAW_DATA_DIR, ASSET_CONFIG['Bitcoin']['filename']), usecols=['timestamp'])['timestamp'].max()
        gold_end = pd.read_csv(os.path.join(RAW_DATA_DIR, ASSET_CONFIG['Gold']['filename']), usecols=['timestamp'])['timestamp'].max()
        res = {'GVZ': fetch_gvz, 'DVOL': fetch_dvol}
        for name, end in (('GVZ', gold_end), ('DVOL', btc_end)):
            print(name, '→', save(name, res[name](), RAW_DATA_DIR, until=end), 'until', end)
    else:
        for k, v in fetch_all().items():
            print(k, 'OK' if v is not None else 'FAILED', '' if v is None else f"{len(v)} rows → {v['timestamp'].max().date()}")
