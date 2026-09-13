# Results and Interpretation (final system)

All numbers are produced by `src/evaluation/backtesting.py` from the frozen data. Raw tables: `results/cv_results.csv` (walk-forward validation), `results/final_test_results.csv` (untouched test), `results/FINAL_RESULTS.md` (all models, all metrics), `results/regime_analysis.csv`, `results/experiments/*.csv`. Figures: `results/figures/`.

**Data:** Yahoo Finance daily, 2018-01 → 2026-09-12 (Bitcoin 3,147 usable days; Gold/Silver 2,156 exchange days).
**Split:** train ≤ 2025-09-10 (BTC 2,750 / metals 1,874 windows) · validation ≤ 2026-02-17 (160 / 109) · **unseen test 2026-02-18 → 2026-09-12** (BTC 207 days, Gold/Silver 143 exchange days). Served models are selected by mean walk-forward RMSE on train+val folds; the test set was evaluated once.
**Close-time rule (final audit):** the metals' macro and High/Low-based features use the previous session's values because Yahoo's GC=F/SI=F close is the 13:30 ET settlement (`docs/METHODOLOGY.md §2`). Section 2 shows what the earlier same-day alignment did.

## 1. Final results table (unseen test set)

| Asset | Model | MAE ($) | RMSE ($) | MAPE % | RMSE (return) | R² (return) | Directional acc. (p) | DM p vs naive |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Bitcoin | Naive (random walk) | 1,066.42 | 1,451.74 | 1.52 | 0.02079 | −0.001 | — | — |
| Bitcoin | **CatBoost (served)** | 1,071.08 | 1,453.76 | 1.53 | 0.02082 | −0.004 | 46.9 % (0.83) | 0.74 |
| Bitcoin | Stacked ensemble | 1,061.34 | 1,442.94 | 1.52 | 0.02065 | +0.013 | 50.2 % (0.50) | 0.44 |
| Gold | Naive (random walk) | 58.26 | 75.76 | 1.29 | 0.01671 | −0.002 | — | — |
| Gold | **CatBoost (served)** | 58.43 | 76.22 | 1.30 | 0.01681 | −0.015 | 51.7 % (0.37) | 0.71 |
| Gold | Random Forest | 58.08 | 75.10 | 1.29 | 0.01659 | +0.012 | 49.0 % (0.63) | 0.57 |
| Silver | Naive (random walk) | 1.79 | 2.37 | 2.49 | 0.03189 | −0.001 | — | — |
| Silver | **GRU (served)** | 1.79 | 2.37 | 2.48 | 0.03189 | −0.001 | 55.2 % (0.12) | 0.99 |
| Silver | Random Forest | 1.76 | 2.33 | 2.44 | 0.03146 | +0.026 | 56.6 % (0.07) | 0.17 |
| Silver | LightGBM | 1.79 | 2.37 | 2.48 | 0.03189 | −0.001 | 58.7 % (0.02) | 0.98 |

Full 10-model tables per asset: `results/FINAL_RESULTS.md`.

**Reading.** For all three assets every model sits at the random-walk floor: the served models' return-RMSE is 0.0–0.6 % *above* the naive forecast (never below it), no Diebold–Mariano test is significant (p = 0.71–0.99), and the R² in return space is ≈ 0. The served models' USD errors — MAPE 1.53 % (Bitcoin), 1.30 % (Gold), 2.48 % (Silver) — are the same as the random walk's (1.52 / 1.29 / 2.49 %) because that is the daily volatility of each asset, not model skill. **The honest answer to the research question is: no model beats the random walk at a one-day horizon for Bitcoin, Gold or Silver in this data.**

**The one p < 0.05 cell.** Silver LightGBM scores 58.7 % directional accuracy (binomial p = 0.02) but has *exactly* the random walk's RMSE (DM p = 0.98): it gets the sign right slightly more often while adding nothing to magnitude. With 27 model-rows in the test table, one cell at p ≈ 0.02 is what chance produces (`docs/LIMITATIONS.md §2`); its walk-forward figure is 53.3 %, and it was not the validation winner. It is not claimed.

## 2. The leak the final audit found — same-day macro features for the metals

The first evaluation of the improved system (`results/archive_sameday_macro_leak/`) reported Silver LightGBM at **62.2 % directional accuracy, R² = +0.044, DM p = 0.002**. That version aligned the S&P 500, VIX, 10-year-yield, DXY and WTI returns *same-day* onto the metals' calendar. Yahoo's daily close for GC=F/SI=F is the **13:30 ET COMEX settlement** (verified against 5-minute bars), so those returns — fixed between 14:30 and 17:00 ET — and the session High/Low contain information from *inside* the target interval.

| Silver, served model on the same 143 unseen days | Dir. acc. (p) | DM p vs naive | R² (return) |
|---|---:|---:|---:|
| same-day macro features (archived, leak) — LightGBM | 62.2 % (0.002) | 0.002 | +0.044 |
| the five macro features removed — LightGBM | 51.7 % (0.37) | 0.21 | +0.015 |
| **previous-day macro + lagged High/Low (final)** — GRU | 55.2 % (0.12) | 0.99 | −0.001 |

Bitcoin was never affected (its bar closes at 00:00 UTC, after every US close — its numbers are identical before and after), and Gold was at the floor either way. The effect was invisible in walk-forward validation (E2 ablation: dropping the macro group changed CV RMSE by −0.04 %), which is itself a lesson: a leak that is small on average can still dominate a single volatile test window in which the metals' contemporaneous correlation with equities/rates doubled. The corrected rule is implemented in `DataCleaner._align_external` / `lag_post_settlement_features`, enforced by two unit tests, and documented in `docs/FEATURES.md`.

## 3. Before vs after (identical unseen days: 2026-02-18 → 2026-07-28)

The 3-year system (`results/archive_3y_final/`) and the final system compared on exactly the same days. Yahoo revised a few historical closes between the two downloads (mean < 0.005 %, max 0.75 % on one Bitcoin day), so each system is also shown against the naive forecast computed on its own data. Source: `results/experiments/before_after.csv`. Note that this script reads the test rows a second time; it compares two frozen systems and selects nothing (`docs/LIMITATIONS.md §2c`).

| Asset | System | Model | MAE ($) | RMSE ($) | MAPE % | RMSE (ret) | R² (ret) | Dir. acc. (p) | DM p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Bitcoin | before (3 y) | CatBoost | 1,096.70 | 1,424.81 | 1.59 | 0.02068 | −0.005 | 47.8 % (0.74) | 0.57 |
| Bitcoin | **after (2018→)** | CatBoost | 1,100.64 | 1,425.57 | 1.59 | 0.02071 | −0.007 | 46.6 % (0.83) | 0.34 |
| Bitcoin | naive (new data) | — | 1,092.36 | 1,420.93 | 1.58 | 0.02064 | −0.000 | — | — |
| Gold | before (3 y) | CatBoost | 61.74 | 80.03 | 1.36 | 0.01752 | −0.041 | 47.7 % (0.72) | 0.15 |
| Gold | **after (2018→)** | CatBoost | 61.07 | 79.54 | 1.35 | 0.01742 | −0.030 | 49.5 % (0.58) | 0.62 |
| Gold | naive (new data) | — | 60.87 | 78.79 | 1.34 | 0.01725 | −0.010 | — | — |
| Silver | before (3 y) | CatBoost | 1.97 | 2.59 | 2.67 | 0.03418 | −0.017 | 48.6 % (0.65) | 0.27 |
| Silver | **after (2018→)** | GRU | 1.95 | 2.57 | 2.65 | 0.03403 | −0.007 | 53.2 % (0.28) | 0.74 |
| Silver | naive (new data) | — | 1.96 | 2.57 | 2.66 | 0.03399 | −0.004 | — | — |

* **Gold:** return-RMSE −0.6 %; **Silver:** −0.4 %; **Bitcoin:** +0.1 %. All within noise: the served models now sit *on* the random-walk line instead of 0.6–1.5 % below it, which is the most a 2.9× larger training set bought. The earlier claim of a 2.8 % Silver improvement was the leak.

## 4. What the improvement phase actually changed

| Change | Evidence | Effect |
|---|---|---|
| 2.9× more training data (2018→) | `E1_data_size.csv`: fixed validation window, same features/params | Bitcoin −1.0 to −1.7 % val RMSE; Gold ±0.4 %; Silver −1.4 % (CatBoost) to +0.7 % (LightGBM) — mostly neutral |
| New features: HAR realised volatilities, EWMA volatility, 20-day momentum | `E2_feature_ablation.csv` | no group moves CV RMSE by more than ±0.4 %; dropping the volatility or momentum group is *neutral-to-slightly-better* for the metals' LightGBM |
| Re-tuning with more data | `results/tuning/*.csv` | the tuner accepts a little more capacity for Bitcoin (31 leaves / depth 7) and the most regularised settings for the metals (depth 3, L2 10, 8–31 leaves, min-leaf 50–100); all configurations within 0.4–0.5 % of each other |
| Stacked ensemble | `results/stacking/*.json` | weights non-zero (BTC: LightGBM 0.39, CatBoost 0.31; Gold: CatBoost 1.18, GRU 0.59; Silver: GRU 0.99, LightGBM 0.55) but the stack does not beat the best single model on test |
| Close-time rule for the metals (final audit) | §2 | removes a spurious significant result |
| Split defined by target dates (exact embargo) | `build_dataset` | correctness for multi-day targets; no effect on the 1-day task |

## 5. Design experiments (validation only) — `results/experiments/experiments_summary.md`

* **E1 data size:** more history helps Bitcoin (−1 to −1.7 % RMSE on the fixed validation window) and is neutral for the metals; kept because it also stabilises tuning.
* **E2 feature ablation:** dropping any one group changes walk-forward RMSE by −0.39 % … +0.05 % — inside noise. The full set is kept because it is cheap and interpretable and no group hurts.
* **E3 horizon:** the 5-day return is not more predictable than the next day (Gold models 0.4–1.0 % better than naive, Bitcoin worse); the served horizon stays 1 day.
* **E4 volatility:** 22-day realised volatility *is* more predictable than the return (Bitcoin Ridge 11.5 % better than persistence), but only ~4 % better than a RiskMetrics EWMA and worse than EWMA for Silver; not productised — kept as a documented experiment.

## 6. Performance across market regimes (test period, served model vs naive) — `results/regime_analysis.csv`

| Asset | Regime | Days | MAE % served | MAE % naive | Direction hits |
|---|---|---:|---:|---:|---:|
| Bitcoin | low / medium / high volatility | 69 / 69 / 69 | 1.30 / 1.64 / 1.65 | 1.29 / 1.62 / 1.66 | 46 / 46 / 48 % |
| Bitcoin | down-trend / up-trend | 96 / 111 | 1.64 / 1.44 | 1.62 / 1.44 | 41 / 52 % |
| Gold | low / medium / high volatility | 48 / 47 / 48 | 1.28 / 1.28 / 1.33 | 1.27 / 1.26 / 1.35 | 52 / 51 / 52 % |
| Silver | low / medium / high volatility | 48 / 47 / 48 | 2.11 / 2.40 / 2.93 | 2.14 / 2.38 / 2.93 | 60 / 45 / 60 % |
| Silver | down-trend / up-trend | 81 / 62 | 2.57 / 2.36 | 2.58 / 2.36 | 59 / 50 % |

Errors scale with volatility for every asset (as they must); the served models neither improve nor degrade relative to the naive forecast in any regime.

## 7. Over-fitting check — `figures/overfitting_gap.png`, `results/train_val_metrics.csv`
Validation RMSE is 0.8× (Bitcoin), 2.1× (Gold) and 2.9× (Silver) the training RMSE, and the random walk shows the same ratios: the gap is regime shift (the validation window is far more volatile than 2018–2025 on average), not memorisation. Early stopping selected 10–82 boosting rounds for the metals' LightGBM/CatBoost (Gold CatBoost 606 at learning-rate 0.01) and 1–9 epochs for the recurrent models; the served Silver GRU predicts with a standard deviation of 0.0009 against a realised 0.032 — it is, in effect, the drift.

## 8. What can be claimed

* *"Under a frozen chronological split with walk-forward model selection and a single evaluation on 143–207 unseen days, none of ten model families beats the random-walk forecast for Bitcoin, Gold or Silver at a one-day horizon (Diebold–Mariano p ≥ 0.71 for the served models; directional accuracy 47–55 %, none significant)."*
* *"The served models' next-day error is 1.3 % (Gold), 1.5 % (Bitcoin) and 2.5 % (Silver) of price — identical to the random walk's, which is the volatility floor of each asset. This number is not 'accuracy'."*
* *"The project's own leakage audit found that a same-day alignment of macro features with the metals' 13:30 ET settlement manufactured a 62 % directional-accuracy result; correcting the alignment removed it."*

Not claimable: any "X % accuracy" derived from MAPE or from R² on price levels; a market-beating edge on any asset; the archived Silver result.
