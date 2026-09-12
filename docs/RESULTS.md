# Results and Interpretation (improved system)

All numbers are produced by `src/evaluation/backtesting.py` from the frozen data. Raw tables: `results/cv_results.csv` (walk-forward validation), `results/final_test_results.csv` (untouched test), `results/FINAL_RESULTS.md` (all models, all metrics), `results/regime_analysis.csv`, `results/experiments/*.csv`. Figures: `results/figures/`.

**Data:** Yahoo Finance daily, 2018-01 → 2026-09-12 (Bitcoin 3,147 usable days; Gold/Silver 2,156 exchange days).
**Split:** train ≤ 2025-09-10 (BTC 2,750 / metals 1,874 windows) · validation ≤ 2026-02-17 (160 / 109) · **unseen test 2026-02-18 → 2026-09-12** (BTC 207 days, Gold/Silver 143 exchange days). Served models are selected by mean walk-forward RMSE on train+val folds; the test set was evaluated once.

## 1. Final results table (unseen test set)

| Asset | Model | MAE ($) | RMSE ($) | MAPE % | R² (return) | Directional acc. (p) | DM p vs naive |
|---|---|---:|---:|---:|---:|---:|---:|
| Bitcoin | Naive (random walk) | 1,066.42 | 1,451.74 | 1.52 | −0.001 | — | — |
| Bitcoin | **CatBoost (served)** | 1,071.08 | 1,453.76 | 1.53 | −0.004 | 46.9 % (0.83) | 0.74 |
| Bitcoin | LightGBM | 1,076.49 | 1,446.18 | 1.54 | +0.010 | 51.2 % (0.39) | 0.79 |
| Bitcoin | Stacked ensemble | 1,061.34 | 1,442.94 | 1.52 | +0.013 | 50.2 % (0.50) | 0.44 |
| Gold | Naive (random walk) | 58.26 | 75.76 | 1.29 | −0.002 | — | — |
| Gold | **LightGBM (served)** | 58.02 | 75.30 | 1.29 | +0.008 | 51.0 % (0.43) | 0.42 |
| Gold | CatBoost | 57.82 | 75.78 | 1.28 | −0.004 | 54.5 % (0.16) | 0.95 |
| Silver | Naive (random walk) | 1.79 | 2.37 | 2.49 | −0.001 | — | — |
| Silver | **LightGBM (served)** | 1.75 | 2.32 | 2.42 | **+0.044** | **62.2 % (0.002)** | **0.002** |
| Silver | Stacked ensemble | 1.75 | 2.32 | 2.42 | +0.043 | 62.2 % (0.002) | 0.023 |
| Silver | Random Forest | 1.76 | 2.34 | 2.44 | +0.025 | 55.9 % (0.09) | 0.16 |

Full 10-model tables per asset: `results/FINAL_RESULTS.md`.

**Reading:** Bitcoin and Gold remain at the random-walk floor (all models within ±0.5 % of the naive RMSE, directional accuracy 47–55 %, no DM test significant). **Silver is the exception**: the served LightGBM has R² = +0.044 in return space, 62.2 % directional accuracy on 143 days (binomial p = 0.002) and a lower squared error than the random walk with Diebold–Mariano p = 0.002. The stacked ensemble reproduces the same result. This is the first statistically significant out-of-sample result in the project.

**Caveat on Silver.** The walk-forward validation directional accuracy of the same model was 52.8 % (RMSE 0.3 % better than naive), so the test period is unusually favourable; the regime table shows the edge is present in every regime of the test window (56–68 % hits in all six slices), but one 143-day window is a single draw and the effect may not persist. It is reported as a genuine but period-specific result, not as a proven edge.

## 2. Before vs after (identical unseen days: 2026-02-18 → 2026-07-28)

The 3-year system (`results/archive_3y_final/`) and the improved system compared on exactly the same days. Yahoo revised a few historical closes between the two downloads (mean < 0.005 %, max 0.75 % on one Bitcoin day), so each system is also shown against the naive forecast computed on its own data. Source: `results/experiments/before_after.csv`.

| Asset | System | Model | MAE ($) | RMSE ($) | MAPE % | RMSE (ret) | R² (ret) | Dir. acc. (p) | DM p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Bitcoin | before (3 y) | CatBoost | 1,096.70 | 1,424.81 | 1.59 | 0.02068 | −0.005 | 47.8 % (0.74) | 0.57 |
| Bitcoin | **after (2018→)** | CatBoost | 1,100.64 | 1,425.57 | 1.59 | 0.02071 | −0.007 | 46.6 % (0.83) | 0.34 |
| Bitcoin | naive (new data) | — | 1,092.36 | 1,420.93 | 1.58 | 0.02064 | −0.000 | — | — |
| Gold | before (3 y) | CatBoost | 61.74 | 80.03 | 1.36 | 0.01752 | −0.041 | 47.7 % (0.72) | 0.15 |
| Gold | **after (2018→)** | LightGBM | 60.62 | 78.39 | 1.34 | 0.01718 | −0.002 | 49.5 % (0.58) | 0.56 |
| Gold | naive (new data) | — | 60.87 | 78.79 | 1.34 | 0.01725 | −0.010 | — | — |
| Silver | before (3 y) | CatBoost | 1.97 | 2.59 | 2.67 | 0.03418 | −0.017 | 48.6 % (0.65) | 0.27 |
| Silver | **after (2018→)** | LightGBM | 1.91 | 2.51 | 2.60 | 0.03322 | +0.041 | 61.3 % (0.011) | 0.006 |
| Silver | naive (new data) | — | 1.96 | 2.57 | 2.66 | 0.03399 | −0.004 | — | — |

* **Gold:** RMSE (return) improved 1.9 %; the served model now marginally beats the naive forecast instead of being 1.5 % worse.
* **Silver:** RMSE (return) improved 2.8 %; from worse-than-naive to significantly better than naive.
* **Bitcoin:** unchanged within noise (the served CatBoost is 0.3 % worse than naive both before and after). On the full test window LightGBM and the stack do slightly better than naive but not significantly, and CatBoost was the validation winner by 0.06 %.

## 3. What produced the improvement

| Change | Evidence | Effect |
|---|---|---|
| 2.9× more training data (2018→) | `E1_data_size.csv`: fixed validation window, same features/params | Bitcoin −1.0 to −1.7 % val RMSE; Gold/Silver ±0.4 % (neutral) |
| New features: HAR realised volatilities (`rv_1d/5d/22d`), EWMA volatility, 20-day momentum | `E2_feature_ablation.csv` | no single group moves CV RMSE by more than ±0.5 %; volatility features matter most for the metals' importance rankings |
| Re-tuning with more data | `results/tuning/*.csv` | tuner now accepts slightly more capacity (15–31 leaves, depth 5–7) but all configurations are within 0.3–2 %; fold std ≈ 20 % of the mean |
| Stacked ensemble with more data | `results/stacking/*.json` | weights no longer collapse to zero (BTC: LightGBM 0.39, CatBoost 0.31; Gold: all four; Silver: LightGBM 1.1, CatBoost 0.47, GRU 0.56) — the base models now carry combinable out-of-sample signal |
| Split defined by target dates (exact embargo) | `build_dataset` | correctness for multi-day targets; no effect on the 1-day task |

## 4. Design experiments (validation only) — `results/experiments/experiments_summary.md`

* **E1 data size:** more history helps Bitcoin (−1 to −1.7 % RMSE on the fixed validation window) and is neutral for the metals; kept because it also stabilises tuning.
* **E2 feature ablation:** dropping any one group changes walk-forward RMSE by −0.15 % … +0.45 % — inside noise. The full set is kept because it is cheap, interpretable and no group hurts; the "return path" group is the most useful for Gold (+0.45 % when removed).
* **E3 horizon:** the 5-day return is not more predictable than the next day (Gold models 0.6–0.8 % better than naive, Bitcoin worse); the served horizon stays 1 day.
* **E4 volatility:** 22-day realised volatility *is* more predictable than the return (Bitcoin Ridge 11.5 % better than persistence), but only ~4 % better than a RiskMetrics EWMA and worse than EWMA for Silver; not productised — kept as a documented experiment.

## 5. Performance across market regimes (test period, served model vs naive) — `results/regime_analysis.csv`

| Asset | Regime | Days | MAE % served | MAE % naive | Direction hits |
|---|---|---:|---:|---:|---:|
| Bitcoin | low / medium / high volatility | 69 / 69 / 69 | 1.30 / 1.64 / 1.65 | 1.29 / 1.62 / 1.66 | 46 / 46 / 48 % |
| Bitcoin | down-trend / up-trend | 96 / 111 | 1.64 / 1.44 | 1.63 / 1.44 | 41 / 52 % |
| Gold | low / medium / high volatility | 48 / 47 / 48 | 1.27 / 1.26 / 1.34 | 1.27 / 1.26 / 1.35 | 48 / 51 / 54 % |
| Silver | low / medium / high volatility | 48 / 47 / 48 | 2.05 / 2.32 / 2.89 | 2.14 / 2.38 / 2.93 | 67 / 64 / 56 % |
| Silver | down-trend / up-trend | 81 / 62 | 2.50 / 2.32 | 2.58 / 2.36 | 64 / 60 % |

Errors scale with volatility for every asset (as they must); the served models neither improve nor degrade relative to the naive forecast in any regime for Bitcoin/Gold, while Silver's advantage holds in every regime.

## 6. Over-fitting check — `figures/overfitting_gap.png`, `results/train_val_metrics.csv`
Validation RMSE is 0.8× (Bitcoin), 2.1× (Gold) and 3.0× (Silver) the training RMSE, and the random walk shows the same ratios: the gap is regime shift (the validation window is far more volatile than 2018–2025 on average), not memorisation. Early stopping selected 10–50 boosting iterations and 1–12 epochs.

## 7. Conclusion
With 8.6 years of daily data and a leakage-free pipeline, the next-day return of Bitcoin and Gold remains indistinguishable from a random walk (consistent with weak-form efficiency), while Silver shows a statistically significant but period-specific edge (62 % directional accuracy, DM p = 0.002 on 143 unseen days). The improvements over the 3-year system are real but small (1.9–2.8 % RMSE on the metals, none on Bitcoin) and came from data volume rather than model complexity. The demonstration value of the system — predict any unseen day, reveal the actual, measure the error, compare ten models against the random walk — is independent of whether the edge persists, and that is the contribution the project claims.
