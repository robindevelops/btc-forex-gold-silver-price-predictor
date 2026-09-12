"""
Evaluation metrics for next-day return forecasting.

Two spaces are reported:
  * return space  — the quantity the models actually predict (log return). R² and the
                    Diebold–Mariano test are only meaningful here.
  * USD space     — P̂_{t+1} = P_t·exp(r̂). MAE/RMSE/MAPE in dollars are what a user sees,
                    but a random walk already scores R²≈0.95 here, so R²(USD) is NOT reported.

"Directional accuracy" = share of days with a non-zero actual move on which
sign(r̂) == sign(r). Days with exactly zero return are excluded (their direction is undefined).
"""
import numpy as np
from scipy import stats


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = y_true != 0
    return float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100) if m.any() else np.nan


def r2(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - np.sum((y_true - y_pred) ** 2) / ss_tot) if ss_tot else np.nan


def directional_accuracy(true_ret, pred_ret):
    """Returns (accuracy %, n_evaluated_days, one-sided binomial p-value vs 50%)."""
    true_ret, pred_ret = np.ravel(true_ret), np.ravel(pred_ret)
    m = true_ret != 0
    n = int(m.sum())
    if n == 0:
        return np.nan, 0, np.nan
    hits = int(np.sum(np.sign(true_ret[m]) == np.sign(pred_ret[m])))
    p = stats.binomtest(hits, n, 0.5, alternative='greater').pvalue
    return 100.0 * hits / n, n, float(p)


def diebold_mariano(true_ret, pred_a, pred_b, h=1):
    """
    Diebold–Mariano test on squared-error loss differential  d_t = e_a² − e_b².
    H0: equal forecast accuracy. Negative statistic ⇒ model A better than model B.
    Uses HAC (Newey–West) variance with h−1 lags, and the Harvey et al. small-sample correction.
    Returns (statistic, two-sided p-value).
    """
    true_ret, pred_a, pred_b = map(np.ravel, (true_ret, pred_a, pred_b))
    d = (true_ret - pred_a) ** 2 - (true_ret - pred_b) ** 2
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return np.nan, np.nan
    d_bar = d.mean()
    gamma0 = np.mean((d - d_bar) ** 2)
    var = gamma0
    for k in range(1, h):
        gk = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        var += 2 * gk
    var = max(var, 1e-18)
    dm = d_bar / np.sqrt(var / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)      # Harvey–Leybourne–Newbold
    p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)


def strategy_backtest(true_ret, pred_ret, cost_bps=10.0, periods_per_year=252):
    """
    Long/flat rule: hold the asset on day t+1 iff r̂_{t+1} > 0. Transaction cost is
    charged whenever the position changes. Returns dict with cumulative strategy return,
    buy-and-hold return, annualised Sharpe (strategy) and number of trades.
    """
    true_ret, pred_ret = np.ravel(true_ret), np.ravel(pred_ret)
    pos = (pred_ret > 0).astype(float)
    trades = np.abs(np.diff(np.concatenate([[0.0], pos])))
    strat = pos * true_ret - trades * cost_bps / 1e4
    cum_strat = float(np.exp(strat.sum()) - 1) * 100
    cum_bh = float(np.exp(true_ret.sum()) - 1) * 100
    sharpe = float(strat.mean() / strat.std() * np.sqrt(periods_per_year)) if strat.std() > 0 else np.nan
    sharpe_bh = float(true_ret.mean() / true_ret.std() * np.sqrt(periods_per_year)) if true_ret.std() > 0 else np.nan
    return {'strategy_return_pct': cum_strat, 'buy_hold_return_pct': cum_bh,
            'strategy_sharpe': sharpe, 'buy_hold_sharpe': sharpe_bh, 'n_trades': int(trades.sum()),
            'time_in_market_pct': float(pos.mean() * 100)}


def evaluate_forecast(true_ret, pred_ret, prev_close, naive_pred_ret=None):
    """
    Full metric set for one model on one split.
        true_ret / pred_ret : real (unscaled) log returns of day t+1
        prev_close          : close of day t (USD)
        naive_pred_ret      : reference forecast for the DM test (default: zero return)
    """
    true_ret, pred_ret, prev_close = map(np.ravel, (true_ret, pred_ret, prev_close))
    true_usd = prev_close * np.exp(true_ret)
    pred_usd = prev_close * np.exp(pred_ret)
    da, n_da, p_da = directional_accuracy(true_ret, pred_ret)
    if naive_pred_ret is None:
        naive_pred_ret = np.zeros_like(true_ret)
    dm, p_dm = diebold_mariano(true_ret, pred_ret, naive_pred_ret)
    out = {
        'RMSE_ret': rmse(true_ret, pred_ret), 'MAE_ret': mae(true_ret, pred_ret), 'R2_ret': r2(true_ret, pred_ret),
        'RMSE_usd': rmse(true_usd, pred_usd), 'MAE_usd': mae(true_usd, pred_usd), 'MAPE_usd': mape(true_usd, pred_usd),
        'DirAcc_pct': da, 'DirAcc_n': n_da, 'DirAcc_pvalue': p_da,
        'DM_stat_vs_naive': dm, 'DM_pvalue': p_dm,
        'pred_std': float(np.std(pred_ret)), 'true_std': float(np.std(true_ret)),
    }
    out.update(strategy_backtest(true_ret, pred_ret))
    return out


# Backwards-compatible names used by older tests
compute_rmse, compute_mae, compute_mape, compute_r2 = rmse, mae, mape, r2


def qlike(true_var, pred_var):
    """QLIKE loss for variance forecasts (Patton 2011): mean( true/pred − ln(true/pred) − 1 ). Lower is better."""
    ratio = np.asarray(true_var) / np.maximum(np.asarray(pred_var), 1e-12)
    return float(np.mean(ratio - np.log(ratio) - 1))


def evaluate_vol_forecast(true_logrv, pred_logrv, naive_logrv):
    """
    Metrics for a volatility forecast expressed in log realised-volatility space.
        RMSE/MAE/R² in log space; MAE and MAPE in volatility (%) space; QLIKE on variances;
        directional accuracy of the *change* in volatility vs the naive persistence forecast;
        Diebold–Mariano test of squared log errors against the naive forecast.
    """
    true_logrv, pred_logrv, naive_logrv = map(np.ravel, (true_logrv, pred_logrv, naive_logrv))
    tv, pv = np.exp(true_logrv), np.exp(pred_logrv)
    dm, p_dm = diebold_mariano(true_logrv, pred_logrv, naive_logrv)
    # direction of the volatility change relative to the last observed volatility (naive)
    da, n_da, p_da = directional_accuracy(true_logrv - naive_logrv, pred_logrv - naive_logrv)
    return {
        'RMSE_log': rmse(true_logrv, pred_logrv), 'MAE_log': mae(true_logrv, pred_logrv), 'R2_log': r2(true_logrv, pred_logrv),
        'MAE_vol_pct': mae(tv * 100, pv * 100), 'MAPE_vol': mape(tv, pv), 'QLIKE': qlike(tv ** 2, pv ** 2),
        'DirAcc_pct': da, 'DirAcc_n': n_da, 'DirAcc_pvalue': p_da,
        'DM_stat_vs_naive': dm, 'DM_pvalue': p_dm, 'pred_std': float(np.std(pred_logrv)), 'true_std': float(np.std(true_logrv)),
    }


def evaluate_task(data, y_real_true, y_real_pred, naive_real, prev_close=None):
    """Dispatch on the task kind stored in the dataset dict."""
    if data['kind'] == 'vol':
        return evaluate_vol_forecast(y_real_true, y_real_pred, naive_real)
    return evaluate_forecast(y_real_true, y_real_pred, prev_close, naive_pred_ret=naive_real)
