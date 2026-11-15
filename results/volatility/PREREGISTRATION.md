# Pre-registration — volatility forecast (#1), uncertainty band (#2), implied volatility (#3)

Written 2026-11-15 BEFORE any of the candidates below were run. The decision rules are fixed here and are not changed
after seeing results. The test split (target dates after 2026-02-17) is read exactly once, by the final step, after
every selection has been made; nothing is selected on it.

## Data

* Frozen dataset as for every other result (`data/raw`, `data/processed`), same split: train ≤ 2025-09-10,
  validation ≤ 2026-02-17, test = remainder.
* New free series, truncated to the frozen dataset's last day:
  * GVZ — CBOE Gold ETF Volatility Index (CBOE daily history CSV), from 2009-09. Published at the US close (16:15 ET),
    i.e. AFTER the 13:30 ET COMEX settlement → used with a one-day lag for Gold and Silver (same rule as every macro
    series, config.EXTERNAL_SAME_DAY). No long free silver implied-volatility series exists; GVZ is tested for Silver as
    a cross-asset proxy.
  * DVOL — Deribit Bitcoin implied-volatility index (public API, daily candles), from 2021-03-24. The candle dated D
    closes at 00:00 UTC of D+1, the same instant as Yahoo's BTC-USD bar D → same-day value for Bitcoin.

## #1 + #3 — volatility forecast

Target (config.TASKS['vol_5d']): log of the realised volatility of the next 5 trading days,
ln sqrt(mean r²_{t+1..t+5}), built by `build_dataset(asset, task='vol_5d')` (exact embargo).

Candidates (all predict the target from information at day t):

| Name | Definition |
|---|---|
| Persistence | ln rv_5d_t (the task's naive forecast) |
| EWMA | ln ewma_vol_t (RiskMetrics λ = 0.94) |
| HAR | OLS of the target on ln rv_1d*, ln rv_5d, ln rv_22d (Corsi 2009, log form; rv_1d floored at its training 5 % quantile) |
| HAR-IV (#3) | HAR + ln IV_t (DVOL for Bitcoin, GVZ for Gold and Silver) |
| Ridge, LightGBM, CatBoost | the project's tabular models on all 28–29 scaled features, default parameters (config.DEFAULT_PARAMS) |

Evaluation: the project's 4-fold expanding walk-forward folds on train+val (`walk_forward_splits`, purge h−1 = 4),
out-of-fold predictions pooled. Metrics: RMSE of log volatility (primary), QLIKE, Diebold–Mariano test.

Decision rule:
1. Among the candidates WITHOUT implied volatility, select the lowest pooled RMSE (log vol) on all out-of-fold days.
2. HAR-IV replaces it only if, on the out-of-fold days where the implied-volatility series exists, it has lower RMSE
   than the step-1 choice with DM p < 0.05.
3. The selected forecast is served only if it beats Persistence on the pooled out-of-fold days with DM p < 0.05.
   Otherwise the dashboard shows no ML volatility forecast for that asset (only the EWMA value, labelled as such).
4. Each asset is decided independently. The chosen model is refit on train+val and evaluated once on the test split
   against Persistence and EWMA; that result is reported whatever it is.

## #2 — the 68 % uncertainty band

Band P_t·exp(r̂ ± k·σ_t), r̂ = the served Combined forecast. Candidates for σ_t (one-day volatility at day t):

| Name | Definition |
|---|---|
| EWMA (served now) | ewma_vol_t |
| HAR-1d | sqrt of the OLS fit of r²_{t+1} on rv_1d², rv_5d², rv_22d² (floored at the training 1 % quantile of r²) |
| GARCH(1,1) | Gaussian GARCH(1,1) fitted by maximum likelihood on training-split returns, filtered forward with fixed parameters |

HAR-1d and GARCH parameters are fitted on the TRAINING split only. k is calibrated exactly as now: the 68.27 % quantile
of |r − r̂|/σ_t over the walk-forward out-of-fold Combined errors whose target lies in the training split.

Decision rule: per asset, a candidate replaces EWMA only if, on the validation split's out-of-fold days (out-of-sample for
both the σ parameters and k), it has (a) lower QLIKE of σ_t² against the squared forecast error and (b) a coverage
|coverage − 68.27 %| no larger than EWMA's. If both candidates qualify, the lower QLIKE wins. Otherwise EWMA stays.
The served band is then scored once on the test split.
