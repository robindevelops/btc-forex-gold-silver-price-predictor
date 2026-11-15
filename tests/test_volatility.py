"""Volatility forecast (#1, #3) and band volatility (#2): no look-ahead, honest fallbacks, served choice = pre-registered rule."""
import os
import json
import numpy as np
import pandas as pd
import pytest

from config import MODELS_DIR, RESULTS_DIR, PROCESSED_DATA_DIR, get_prefix
from src.volatility import implied_vol as ivm
from src.volatility.models import HARLog, GARCH11, MIN_FIT_ROWS

ASSETS = ('Bitcoin', 'Gold', 'Silver')
HAVE_VOL = all(os.path.exists(os.path.join(MODELS_DIR, f'{get_prefix(a)}_vol_5d.json')) for a in ASSETS)


def _write_iv(tmp_path, name, dates, values):
    pd.DataFrame({'timestamp': pd.to_datetime(dates), 'iv': values}).to_csv(tmp_path / ivm.IV_FILES[name], index=False)


def test_metals_use_previous_day_gvz_and_bitcoin_same_day_dvol(tmp_path):
    _write_iv(tmp_path, 'GVZ', ['2026-03-02', '2026-03-03', '2026-03-04'], [10.0, 20.0, 30.0])
    _write_iv(tmp_path, 'DVOL', ['2026-03-02', '2026-03-03', '2026-03-04'], [50.0, 60.0, 70.0])
    days = pd.to_datetime(['2026-03-03', '2026-03-04', '2026-03-05'])
    gold = ivm.align_iv('Gold', days, str(tmp_path)).values
    btc = ivm.align_iv('Bitcoin', days, str(tmp_path)).values
    assert list(gold) == [10.0, 20.0, 30.0]              # GVZ is published after the 13:30 settlement → value of the day before
    assert list(btc[:2]) == [60.0, 70.0] and np.isnan(btc[2])   # DVOL known at the BTC close; no forward fill


def test_stale_gvz_is_missing_not_carried_forward(tmp_path):
    _write_iv(tmp_path, 'GVZ', ['2026-03-02'], [10.0])
    assert np.isnan(ivm.align_iv('Gold', pd.to_datetime(['2026-03-20']), str(tmp_path)).values[0])


def test_har_refuses_to_fit_on_too_few_rows():
    idx = pd.date_range('2020-01-01', periods=MIN_FIT_ROWS + 50)
    rng = np.random.default_rng(0)
    feats = pd.DataFrame({c: rng.uniform(0.005, 0.05, len(idx)) for c in ('rv_1d', 'rv_5d', 'rv_22d')}, index=idx)
    y = np.log(feats['rv_5d'].values)
    small = HARLog().fit(feats, idx[:MIN_FIT_ROWS - 1], y[:MIN_FIT_ROWS - 1])
    assert small.coef is None and np.isnan(small.predict(feats, idx[-5:])).all()      # no forecast, never a zero forecast
    full = HARLog().fit(feats, idx, y)
    assert np.isfinite(full.predict(feats, idx[-5:])).all()


def test_garch_sigma_uses_only_past_returns():
    r = np.random.default_rng(1).normal(0, 0.02, 600)
    g = GARCH11().fit(r[:400])
    assert 0 < g.alpha + g.beta < 1
    s1 = g.sigma(r)
    r2 = r.copy(); r2[500:] *= 10                         # change the future
    s2 = g.sigma(r2)
    assert np.allclose(s1[:500], s2[:500]) and not np.allclose(s1[500:], s2[500:])


@pytest.mark.skipif(not HAVE_VOL, reason="volatility models not evaluated yet")
@pytest.mark.parametrize('asset', ASSETS)
def test_served_volatility_method_follows_the_preregistered_rule(asset):
    wf = pd.read_csv(os.path.join(RESULTS_DIR, 'volatility', 'vol_walkforward.csv'))
    wf = wf[wf['asset'] == asset]
    allw = wf[wf['days'] == 'all out-of-fold'].set_index('candidate')
    step1 = allw.drop(index='HAR-IV')['RMSE_log'].idxmin()
    with open(os.path.join(MODELS_DIR, f'{get_prefix(asset)}_vol_5d.json')) as f:
        art = json.load(f)
    assert art['method'] in (step1, 'HAR-IV')
    assert art['served'] == (art['walkforward']['DM_p'] < 0.05 and art['walkforward']['RMSE_vs_ref_pct'] < 0)


@pytest.mark.skipif(not HAVE_VOL, reason="volatility models not evaluated yet")
@pytest.mark.parametrize('asset', ASSETS)
def test_volatility_forecast_ignores_rows_after_the_forecast_day(asset):
    from src.inference.prediction import _window
    from src.volatility.forecast import forecast_volatility
    feats = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{get_prefix(asset)}_features.csv'), index_col='timestamp', parse_dates=True)
    day = feats.index[feats.index.get_indexer([pd.Timestamp('2026-05-12')], method='pad')[0]]
    full = forecast_volatility(asset, feats, day, live=False, window=_window(asset, feats, day)[:2])
    cut = feats.loc[:day]
    trunc = forecast_volatility(asset, cut, day, live=False, window=_window(asset, cut, day)[:2])
    assert np.isclose(full['daily_vol_pct'], trunc['daily_vol_pct']) and 0.1 < full['daily_vol_pct'] < 20
    assert full['regime'] in ('low', 'normal', 'high')
