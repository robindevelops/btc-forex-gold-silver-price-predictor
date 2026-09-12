# Results and Interpretation

All numbers are produced by `src/evaluation/backtesting.py` from the frozen data and are reproducible with `make pipeline`. Raw tables: `results/cv_results.csv` (walk-forward validation), `results/final_test_results.csv` (untouched test), `results/FINAL_RESULTS.md` (all models, all metrics). Figures: `results/figures/`.

Test period for all assets: **2026-02-18 → 2026-07-28** (Bitcoin 161 days, Gold/Silver 111 exchange days). Served models were selected by mean walk-forward RMSE (return space) on the train+val folds; the test set was evaluated once.

## Final results table (test set)

| Asset | Model | MAE ($) | RMSE ($) | MAPE % | R² (return) | Directional Acc. (p) | DM p vs naive |
|---|---|---:|---:|---:|---:|---:|---:|
| Bitcoin | Naive (random walk) | 1,093.48 | 1,421.12 | 1.58 | −0.000 | — | — |
| Bitcoin | **CatBoost (served)** | 1,096.70 | 1,424.81 | 1.59 | −0.005 | 47.8 % (0.74) | 0.57 |
| Bitcoin | Random Forest | 1,094.69 | 1,414.95 | 1.59 | +0.007 | 52.8 % (0.26) | 0.60 |
| Bitcoin | GRU | 1,210.18 | 1,534.17 | 1.75 | −0.182 | 52.2 % (0.32) | 0.01 (worse) |
| Gold | Naive (random walk) | 60.97 | 78.84 | 1.34 | −0.010 | — | — |
| Gold | **CatBoost (served)** | 61.74 | 80.03 | 1.36 | −0.041 | 47.7 % (0.72) | 0.15 |
| Gold | LightGBM | 62.00 | 80.51 | 1.37 | −0.054 | 49.5 % (0.58) | 0.05 (worse) |
| Gold | GRU | 63.45 | 81.48 | 1.40 | −0.072 | 45.9 % (0.83) | 0.07 |
| Silver | Naive (random walk) | 1.95 | 2.57 | 2.65 | −0.004 | — | — |
| Silver | **CatBoost (served)** | 1.97 | 2.59 | 2.67 | −0.017 | 48.6 % (0.65) | 0.27 |
| Silver | LSTM | 2.09 | 2.72 | 2.83 | −0.129 | 49.5 % (0.58) | 0.08 |
| Silver | GRU | 2.13 | 2.75 | 2.89 | −0.166 | 45.0 % (0.87) | 0.05 (worse) |

(Full 10-model tables per asset in `results/FINAL_RESULTS.md`.)

## Experiment 1 — Baselines
The zero-return random walk is the strongest or joint-strongest forecast on every asset, in both validation and test. The historical-mean (drift) forecast is within 0.2 % of it. ARIMA — order (1,0,0) for Bitcoin, (2,0,2) Gold, (3,0,2) Silver, chosen by AIC — is *worse* than the random walk on the test set (DM p = 0.30 / 0.03 / 0.01), i.e. the small autocorrelations it fits in-sample do not persist.

## Experiment 2 — The original models, evaluated correctly
`docs/AUDIT_SUMMARY.md`: once the price-reconstruction bug and the synthetic weekend rows were removed, the original GRU/LightGBM/stack had negative return-space R² on every asset and the served stack was a constant predictor. The originally reported "R² ≈ 0.94" was the random-walk illusion on price levels.

## Experiment 3 — Tuned models under walk-forward validation (`figures/model_comparison_cv.png`)
* On the 4 expanding folds, the regularised tabular models (Ridge, LightGBM, CatBoost) are within ±0.5 % of the naive RMSE for Bitcoin and *marginally* better than naive for Gold (0.01312 vs 0.01326, −1.1 %) and Silver (0.02797 vs 0.02805, −0.3 %). Fold-to-fold standard deviations (±0.0026 BTC, ±0.0041 Gold, ±0.0150 Silver) are an order of magnitude larger than these differences, so **no model is distinguishable from the random walk on validation**.
* The tuner consistently chose the most regularised configurations (LightGBM: 4 leaves; CatBoost: depth 3–5, learning-rate 0.01; Ridge: α = 100; early stopping at 10 iterations for Bitcoin and Silver, 67 for Gold). The models learned that the best-generalising function is approximately a constant drift — the served CatBoost's test predictions have a standard deviation of 0.0001–0.0003 against a true return standard deviation of 0.017–0.034.
* GRU and LSTM are the worst models on every asset in validation (R² −0.10 to −0.22) despite being reduced to a single 16–32-unit layer with dropout and early stopping: with 500–700 windows they fit noise.

## Experiment 4 — Untouched test set (`figures/model_comparison_test.png`, `figures/<asset>_actual_vs_predicted.png`)
* **No model beats the random walk by a statistically significant margin** (all Diebold–Mariano p > 0.05 in the "better" direction). The served CatBoost is 0.3 % (BTC), 1.5 % (Gold) and 0.6 % (Silver) *worse* than naive on return-RMSE — inside the noise.
* GRU on Bitcoin and Silver, ARIMA on Gold and Silver, and LightGBM on Gold are *significantly worse* than the random walk (DM p ≤ 0.05).
* Directional accuracy is 42–53 % for every model on every asset, and no value is significant (smallest binomial p = 0.26, Random Forest on Bitcoin, 52.8 % of 161 days). Random Forest on Bitcoin also has the best test R² (+0.007) and the only positive strategy return (+3.1 % vs −6.1 % buy-and-hold) — but it was **not** the validation winner (0.02555 CV RMSE, 7th of 9), so treating it as the result would be selection on the test set.
* The long/flat strategy of the served model equals the drift forecast (always long), so it reproduces buy-and-hold minus costs; the LSTM's flat/long switching happens to limit losses on Gold (+0.2 % vs −17.6 %) and Silver (−9.1 % vs −21.7 %), with RMSE that is worse than naive — a reminder that direction and magnitude are different tasks.

## Experiment 5 — Analysis
* **Over-fitting** (`figures/overfitting_gap.png`): validation RMSE is 1.0× (BTC), 2.0× (Gold) and 3.0× (Silver) the training RMSE — but the random walk shows exactly the same ratio. The gap is regime shift (the validation/test windows are far more volatile than training), not memorisation; the phase-A models (Ridge α = 100, depth-3 trees, 10–70 boosting iterations, 2–13 epochs) have almost no capacity to memorise.
* **Feature importance** (`figures/<asset>_feature_importance.png`): the boosted models lean on volatility-regime features (`bb_width`, `atr_norm`, `volatility_30d`), trend-distance ratios (`ema30_ratio`, `macd_hist_norm`) and, for Bitcoin, `fear_greed` and `sp500_return`. These are the features that shift the conditional *mean* only marginally; their real information is about the conditional *variance*, which a point forecast cannot exploit (see Future Work).
* **Stacked ensemble** (`results/stacking/`): with non-negative Ridge on out-of-fold predictions the weights collapse to **0.0 for all four base models on Bitcoin** (intercept only) and to small weights (0.04–0.16) on Gold/Silver. The stack cannot find a combination of base models with out-of-sample signal — an independent confirmation of the conclusion.
* **Loss curves** (`figures/<asset>_loss_curves.png`): validation loss reaches its minimum within 2–13 epochs and then rises while training loss keeps falling — textbook over-fitting that early stopping catches.

## Conclusion
At a one-day horizon, with ~700–1,000 daily observations and price/volume/macro/sentiment features, **neither classical time-series models, regularised tree ensembles nor recurrent networks produce forecasts that are statistically distinguishable from the random walk** for Bitcoin, Gold or Silver. The tuned, validation-selected models converge to the unconditional drift, and every model that deviates from it (ARIMA, GRU, LSTM) is *significantly worse*. This is consistent with weak-form market efficiency and with the reproducibility literature on financial deep learning. The project's contribution is the leakage-audited benchmark, the statistical testing, and a demonstration system that reports its own uncertainty honestly — not a forecasting edge.

## What would change the conclusion (Future Work)
Longer history (2018→, 3× more data), volatility (GARCH-type) targets where the features clearly carry information, multi-day horizons, probabilistic forecasts (quantile regression) whose calibration can be scored, intraday data, and regime-aware models.
