"""
σ_t of the served 68 % band P_t·exp(r̂ ± k·σ_t), per asset, as selected by src/volatility/evaluate.py --part band
(results/volatility/PREREGISTRATION.md, #2). Stored in data/models/<prefix>_band_vol.json:

    EWMA        σ_t = ewma_vol_t (RiskMetrics, λ = 0.94) — the original band, kept unless a candidate wins on validation
    GARCH(1,1)  σ_t = GARCH one-step forecast, parameters fitted on training-split returns, filtered with fixed parameters
    HAR-1d      σ_t = sqrt of a HAR regression of r²_{t+1} on squared realised volatilities (training split)

Without the artefact (before the evaluation has run) σ_t is the EWMA, i.e. the original behaviour.
"""
import os
import json
import numpy as np
import pandas as pd

from config import MODELS_DIR, get_prefix
from src.volatility.models import GARCH11, HAR1d

EWMA_FEATURE = 'ewma_vol'
RULE_TXT = {'EWMA': 'σ_t = EWMA volatility (RiskMetrics, λ = 0.94) at day t',
            'GARCH(1,1)': 'σ_t = GARCH(1,1) one-day volatility forecast at day t (parameters fitted on the training split)',
            'HAR-1d': 'σ_t = HAR one-day volatility forecast at day t (fitted on the training split)'}


def band_artefact(asset):
    path = os.path.join(MODELS_DIR, f'{get_prefix(asset)}_band_vol.json')
    if not os.path.exists(path):
        return {'sigma': 'EWMA'}
    with open(path) as f:
        return json.load(f)


def band_method(asset):
    return band_artefact(asset)['sigma']


def band_rule(asset):
    return f"68 % band P_t · exp(r̂ ± k·σ_t), {RULE_TXT[band_method(asset)]}, k calibrated on walk-forward errors of the training period"


def band_sigma(asset, feats, dates, method=None):
    """σ_t for each day in `dates`, using only `feats` rows up to that day (GARCH filters the return path up to t)."""
    art = band_artefact(asset)
    method = method or art['sigma']
    dates = pd.DatetimeIndex(dates)
    if method == 'EWMA':
        return feats[EWMA_FEATURE].reindex(dates).values
    if method == 'GARCH(1,1)':
        r = feats['log_return']
        return pd.Series(GARCH11.from_dict(art['garch']).sigma(r.values), index=r.index).reindex(dates).values
    if method == 'HAR-1d':
        return HAR1d.from_dict(art['har_1d']).sigma(feats, dates)
    raise ValueError(f"unknown band volatility '{method}'")
