"""Metric definitions."""
import numpy as np
from src.utils.metrics import directional_accuracy, diebold_mariano, r2, rmse, strategy_backtest, evaluate_forecast


def test_directional_accuracy_excludes_flat_days_and_counts_hits():
    true = np.array([0.01, -0.02, 0.0, 0.03, -0.01])
    pred = np.array([0.02, 0.01, 0.0, 0.01, -0.03])
    acc, n, p = directional_accuracy(true, pred)
    assert n == 4 and np.isclose(acc, 75.0) and 0 <= p <= 1


def test_zero_forecast_has_undefined_direction():
    true = np.random.default_rng(1).normal(size=100)
    acc, n, p = directional_accuracy(true, np.zeros(100))
    assert acc == 0.0            # sign(0) never equals ±1 → reported as '—' in tables


def test_r2_of_mean_forecast_is_zero_and_rmse_matches():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(r2(y, np.full(4, 2.5)), 0.0)
    assert np.isclose(rmse(y, y + 1), 1.0)


def test_diebold_mariano_prefers_better_forecast():
    rng = np.random.default_rng(0)
    true = rng.normal(0, 0.02, 400)
    good = true + rng.normal(0, 0.005, 400)
    bad = np.zeros(400)
    stat, p = diebold_mariano(true, good, bad)
    assert stat < 0 and p < 0.01


def test_strategy_backtest_long_only_matches_buy_and_hold_when_always_long():
    true = np.array([0.01, -0.005, 0.02])
    r = strategy_backtest(true, np.ones(3), cost_bps=0)
    assert np.isclose(r['strategy_return_pct'], r['buy_hold_return_pct'])


def test_evaluate_forecast_keys():
    true = np.random.default_rng(2).normal(0, 0.02, 50)
    out = evaluate_forecast(true, np.zeros(50) + 0.001, np.full(50, 100.0))
    for k in ('RMSE_ret', 'MAE_ret', 'R2_ret', 'RMSE_usd', 'MAE_usd', 'MAPE_usd', 'DirAcc_pct', 'DM_pvalue', 'strategy_return_pct'):
        assert k in out
