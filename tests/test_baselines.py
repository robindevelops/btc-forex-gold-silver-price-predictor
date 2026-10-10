"""The fixed moving-average baselines (MA-5, Mean-20d): definition, no look-ahead, and agreement with the stored results."""
import os
import numpy as np
import pandas as pd
import pytest
from config import PROCESSED_DATA_DIR, RESULTS_DIR, ASSETS, get_prefix
from src.models.registry import rolling_baseline_forecast, ROLLING_BASELINES


def _features(prices):
    p = pd.Series(prices, index=pd.bdate_range('2024-01-01', periods=len(prices)))
    r = np.log(p / p.shift(1))
    return pd.DataFrame({'price': p, 'log_return': r, 'return_20d': r.rolling(20).sum()})


def _prices(n=80, seed=0):
    return 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, n)))


def test_ma5_and_mean20d_definitions():
    f = _features(_prices())
    days = f.index[30:45]
    ma, m20 = (rolling_baseline_forecast(b, f, days) for b in ('MA-5', 'Mean-20d'))
    for i, t in enumerate(days):
        pos = f.index.get_loc(t)
        assert np.isclose(ma[i], np.log(f['price'].iloc[pos - 4:pos + 1].mean() / f['price'].iloc[pos]))   # next close = mean of last 5
        assert np.isclose(m20[i], f['log_return'].iloc[pos - 19:pos + 1].mean())                          # mean of last 20 returns


@pytest.mark.parametrize('name', ROLLING_BASELINES)
def test_rolling_baselines_use_only_rows_up_to_day_t(name):
    p = _prices()
    t = 50
    q = p.copy(); q[t + 1:] *= 1.37                                  # rewrite the whole future
    a = rolling_baseline_forecast(name, _features(p), _features(p).index[[t]])
    b = rolling_baseline_forecast(name, _features(q), _features(q).index[[t]])
    assert np.isclose(a, b)


def test_rolling_baseline_refuses_days_without_enough_history():
    f = _features(_prices())
    with pytest.raises(ValueError):
        rolling_baseline_forecast('MA-5', f, f.index[:3])
    with pytest.raises(ValueError):
        rolling_baseline_forecast('Unknown', f, f.index[30:31])


HAVE_RESULTS = (all(os.path.exists(os.path.join(PROCESSED_DATA_DIR, f'{get_prefix(a)}_test_scaled.csv')) for a in ASSETS)
                and os.path.exists(os.path.join(RESULTS_DIR, 'final_test_results.csv')))


@pytest.mark.skipif(not HAVE_RESULTS, reason="processed data / evaluation results not present")
@pytest.mark.parametrize('asset', ASSETS)
def test_stored_baseline_rows_match_the_definition(asset):
    """The evaluation's MA-5 / Mean-20d rows are exactly the fixed rules applied on the test days (same days as every model)."""
    from src.data.preprocessing import build_dataset
    from src.utils.metrics import rmse
    te = pd.read_csv(os.path.join(RESULTS_DIR, 'final_test_results.csv'))
    rows = te[(te['asset'] == asset) & te['model'].isin(ROLLING_BASELINES)].set_index('model')
    if len(rows) < len(ROLLING_BASELINES):
        pytest.skip("evaluation not re-run with the moving-average baselines")
    d = build_dataset(asset)
    for b in ROLLING_BASELINES:
        r = rolling_baseline_forecast(b, d['features'], d['dates_test'])
        assert np.isclose(rows.loc[b, 'RMSE_ret'], rmse(d['y_real_test'], r))
        assert int(rows.loc[b, 'n_test']) == len(d['y_real_test'])
        assert rows.loc[b, 'RMSE_ret'] > te[(te['asset'] == asset) & (te['model'] == 'Naive')]['RMSE_ret'].iloc[0]   # worse than the random walk
