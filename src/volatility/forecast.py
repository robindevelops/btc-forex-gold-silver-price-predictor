"""
Served volatility (risk) forecast: the expected average daily move over the next 5 trading days.

The method per asset was selected by src/volatility/evaluate.py under results/volatility/PREREGISTRATION.md and is stored in
data/models/<prefix>_vol_5d.json (Bitcoin: Ridge on the scaled features; Gold: HAR + GVZ implied volatility;
Silver: EWMA). Every forecast carries its held-out test result against the persistence and EWMA forecasts.
"""
import os
import json
import numpy as np
import pandas as pd

from config import MODELS_DIR, RAW_DATA_DIR, RAW_LIVE_DIR, get_prefix
from src.volatility.implied_vol import align_iv, IV_FOR_ASSET
from src.volatility.models import HARLog

TASK = 'vol_5d'


def vol_artefact(asset):
    path = os.path.join(MODELS_DIR, f'{get_prefix(asset)}_{TASK}.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No volatility model for {asset}: run `python src/volatility/evaluate.py --part vol`")
    with open(path) as f:
        return json.load(f)


def _regime(sigma, thresholds):
    lo, hi = thresholds
    return 'low' if sigma <= lo else ('high' if sigma > hi else 'normal')


def forecast_volatility(asset, feats, as_of_ts, live, window=None):
    """
    σ̂ (decimal, per day) for the 5 trading days after `as_of_ts`, from `feats` rows up to as_of_ts only.
    `live` selects where the implied-volatility series is read from (data/raw_live vs the frozen data/raw).
    `window` = (Xseq, Xt) scaled input already built by the caller (needed only for a tabular model).
    """
    art = vol_artefact(asset)
    method = art['method']
    if method == 'Persistence':
        log_sig = float(np.log(max(feats.loc[as_of_ts, 'rv_5d'], 1e-6)))
    elif method == 'EWMA':
        log_sig = float(np.log(max(feats.loc[as_of_ts, 'ewma_vol'], 1e-6)))
    elif method in ('HAR', 'HAR-IV'):
        m = HARLog.from_dict(art['model'])
        iv = None
        if m.with_iv:
            iv = align_iv(asset, [as_of_ts], RAW_LIVE_DIR if live else RAW_DATA_DIR).values
            if not np.isfinite(iv).all():
                raise ValueError(f"{IV_FOR_ASSET[asset]} implied volatility for {as_of_ts.date()} is not available "
                                 "(sync live data); no volatility forecast is shown rather than a stale one")
        log_sig = float(m.predict(feats, [as_of_ts], iv)[0])
    else:                                                             # a tabular model on the scaled feature window
        from src.models.registry import load_trained
        if window is None:
            raise ValueError("tabular volatility model needs the scaled feature window")
        model = load_trained(art['model']['name'], asset, TASK)
        z = float(np.ravel(model.predict(*window))[0])
        log_sig = z * art['model']['target_std'] + art['model']['target_mean']
    if not np.isfinite(log_sig):
        raise ValueError(f"volatility forecast for {asset} is not finite")
    sigma = float(np.exp(log_sig))
    ewma = float(feats.loc[as_of_ts, 'ewma_vol'])
    return {'method': method, 'is_ml': method not in ('Persistence', 'EWMA'), 'served': bool(art['served']),
            'horizon_days': art['horizon_days'], 'as_of_date': str(as_of_ts.date()),
            'daily_vol_pct': sigma * 100, 'five_day_vol_pct': sigma * np.sqrt(art['horizon_days']) * 100,
            'regime': _regime(sigma, art['regime_thresholds_daily_vol']),
            'train_median_daily_vol_pct': art['train_median_daily_vol'] * 100, 'ewma_daily_vol_pct': ewma * 100,
            'iv_series': art.get('iv_series'), 'decision': art['decision'],
            'test': {k: art['test'].get(k) for k in ('n', 'RMSE_log', 'RMSE_vs_ref_pct_vs_persistence', 'DM_p_vs_persistence',
                                                     'RMSE_vs_ref_pct_vs_ewma', 'DM_p_vs_ewma')}}


def forecast_volatility_safe(asset, feats, as_of_ts, live, window=None):
    """Never raises: returns (result, None) or (None, reason) so a missing IV download cannot break the price forecast."""
    try:
        return forecast_volatility(asset, feats, as_of_ts, live, window), None
    except Exception as e:
        return None, str(e)
