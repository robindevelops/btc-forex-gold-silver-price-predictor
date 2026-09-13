# Methodology

This document is the technical reference for the FYP report (Chapter 3) and the viva.
Every statement here maps to a file in `src/`.

## 1. Problem definition

**Task.** Forecast the next trading day's price move for three assets — Bitcoin (BTC-USD), Gold (GC=F front-month futures) and Silver (SI=F) — and quantify whether machine-learning models add anything beyond the random-walk assumption.

**Target.** The next-day *log return*

```
y_t = ln( P_{t+1} / P_t )
```

where `P_t` is the close on trading day `t`. A price forecast is derived afterwards as `P̂_{t+1} = P_t · exp(ŷ_t)` (`src/utils/inverse_transform.py`).

**Why return, not price.** Prices are non-stationary (ADF p ≈ 0.7–0.98, `results/eda_summary.csv`); returns are stationary (p < 1e-14). A model that predicts price levels can reach R² ≈ 0.95 by copying yesterday's price, which says nothing about skill. Predicting the return makes the random-walk baseline (ŷ = 0) explicit and makes every comparison honest.

**Horizon.** 1 trading day. Bitcoin trades every calendar day; Gold/Silver on exchange days.

**Hypothesis tested.** H0: no model has lower expected squared error than the zero-return (random-walk) forecast. Tested with the Diebold–Mariano statistic on the untouched test set.

## 2. Data

| Series | Source | Ticker | Used for |
|---|---|---|---|
| Bitcoin OHLCV | Yahoo Finance | `BTC-USD` | target + features |
| Gold futures OHLCV | Yahoo Finance | `GC=F` | target + features |
| Silver futures OHLCV | Yahoo Finance | `SI=F` | target + features |
| US Dollar Index | Yahoo Finance | `DX-Y.NYB` | metals features |
| WTI crude | Yahoo Finance | `CL=F` | metals features |
| 10-year Treasury yield | Yahoo Finance | `^TNX` | metals features |
| S&P 500 | Yahoo Finance | `^GSPC` | all assets |
| VIX | Yahoo Finance | `^VIX` | all assets |
| Crypto Fear & Greed | alternative.me | — | Bitcoin |

Stored history: **2018-01-01 → 2026-09-12** (BTC 3,177 raw / 3,147 usable days; Gold/Silver 2,186 raw / 2,156 usable exchange days), downloaded with `config.DATA_START_DATE = 2018-01-01`. The earlier 3-year version (2023-07 → 2026-07) is kept in `data/raw_3y_backup/` and its results in `results/archive_3y_final/` for the before/after comparison.

### Cleaning (`src/data/preprocessing.py::DataCleaner.clean_data`)
1. Drop duplicate timestamps, sort chronologically, drop non-positive prices.
2. **Calendar.** Bitcoin: reindex to every calendar day, forward-fill genuine gaps (0 in the stored data). Gold/Silver: **keep exchange days only** — no synthetic weekend rows (the previous pipeline forward-filled weekends, creating 30 % fake zero-return samples).
3. Compute the log return.
4. Add features (§3), then drop the indicator warm-up period (30 rows).
5. Validate: no NaN/inf, ≥ 100 rows.

### Alignment of external series — close-time rule
Each external series' daily log return is computed on **its own** calendar and then aligned to the asset's calendar by `DataCleaner._align_external`. The rule is *"only values known at the moment `P_t` is observed"*, and that moment differs per asset:

| Asset | When Yahoo's daily close `P_t` is fixed | S&P 500 / VIX / 10-y yield / DXY / WTI close | Macro value used at row *t* |
|---|---|---|---|
| BTC-USD | 00:00 UTC = 19:00–20:00 ET (after the US close) | 16:00 / 16:15 / ~15:00 / 17:00 / 14:30 ET | **same day** (forward-filled) |
| GC=F, SI=F | **13:25–13:30 ET COMEX settlement** | same as above — all *after* 13:30 | **previous trading day** (latest external date strictly before *t*) |

The settlement fact was verified against 5-minute bars of the December-2026 contracts (mean |daily close − 13:30 price| = 0.04 % gold / 0.09 % silver, versus 0.3–1 % against the 17:00 Globex close). Two consequences for the metals, both implemented in `preprocessing.py` and enforced by `tests/test_features.py`:

1. A same-day macro return would contain 1–3.5 hours of price movement that happens *after* `P_t` is fixed, i.e. *inside* the target interval `ln(P_{t+1}/P_t)`. The first version of this project had exactly that leak; it produced an apparent 62 % directional accuracy for Silver that disappeared when the macro features were lagged (`docs/RESULTS.md §2`).
2. Yahoo's futures High/Low span the whole Globex session (past the settlement on ~40 % of days), so the features built from them — `hl_range`, `atr_norm`, `ADX` (`config.POST_SETTLEMENT_FEATURES`) — are lagged one session for the metals.

Gold's same-day return remains a same-day feature for Silver: both settle in the same 13:25–13:30 window (the 5-minute difference is negligible at a daily horizon). Fear & Greed is published at 00:00 UTC for the day, before the Bitcoin bar for that day closes.

## 3. Features

See `docs/FEATURES.md` for the full table with formulas. Design rules:

* **Backward-looking only** — every feature at row *t* uses information available when `P_t` is observed (`tests/test_features.py::test_no_lookahead_in_any_feature` perturbs the last row and asserts earlier rows are unchanged, for the crypto and the futures path; `test_commodity_external_features_are_previous_day_values` and `test_commodity_high_low_features_are_lagged_one_session` enforce the close-time rule above).
* **Stationary only** — raw levels (open/high/low/close/volume, EMA, Bollinger bands, ATR, MACD) are kept in `*_features.csv` for charts but are never model inputs (`config.LEVEL_COLUMNS`). They are converted to ratios (`price/EMA − 1`, `%B`, `ATR/price`, …). Reason: the Gold/Silver test period lies entirely above the training price range, so a level feature would be out-of-distribution for every test day.
* **Asset-specific** — Bitcoin gets sentiment (Fear & Greed), day-of-week (7-day market) and volume change; metals get macro drivers (DXY, oil, yields) as previous-day returns; Silver additionally gets gold's same-day return.

Bitcoin: 29 features · Gold: 28 · Silver: 29 (HAR realised volatilities, EWMA volatility and 20-day momentum were added in the improvement phase).

## 4. Split, scaling, sequences

```
train      2018-03 → 2025-09-10          (BTC 2,750 windows, metals 1,874)
validation 2025-09-11 → 2026-02-17       (160 / 109)
test       2026-02-18 → 2026-09-12       (207 / 143)   ← touched ONCE
```

* Chronological by calendar date (`config.TRAIN_END`, `config.VAL_END`), identical for all assets, **no shuffling** — a random split would put tomorrow in the training set of today.
* The split is defined by the dates the **target** covers: a training target must end on or before `TRAIN_END`, a validation target must start after `TRAIN_END` and end on or before `VAL_END`, a test target must start after `VAL_END`. For multi-day experimental targets this is an exact purge/embargo; walk-forward folds additionally purge the last h−1 training samples before each validation block.
* The target is standardised with the training mean/std (`data['fwd']` / `data['inv']`); tree models are unaffected, recurrent models train on a unit-variance target.
* `MinMaxScaler` fitted on the **training rows only** and applied to validation/test. Test-period feature values may fall outside [0, 1]; that is expected and proves the scaler did not see them.
* Windows of `SEQ_LEN = 30` rows feed the recurrent models; the last row of each window feeds the tabular models (`build_dataset(asset, task=…)` in `preprocessing.py`). Both see the same samples and predict the same target, so every model is compared on identical days. `build_dataset` also serves the experimental tasks (`return_5d`, `vol_5d`, `vol_22d`) and a `train_start` option for the data-size experiment.
* Windows are cut from the concatenated train|val|test frame so the first validation/test samples can look back into earlier rows — this uses only past data and loses no test days.

## 5. Baselines and models (`src/models/registry.py`)

| Family | Model | Input | Notes |
|---|---|---|---|
| Baseline | **Naive** | — | ŷ = 0 ⇔ tomorrow's price = today's (random walk); for the volatility task: last realised volatility (persistence). |
| Baseline | Naive-Mean | — | ŷ = mean training return (drift). |
| Baseline | ARIMA(p,0,q) | return series | order by AIC on train, one-step walk-forward. |
| Baseline | EWMA | volatility task only | RiskMetrics λ = 0.94. |
| Linear ML | Ridge | 28–29 features (day *t*) | L2 regularised linear regression. |
| Tree ML | Random Forest | same | bagged trees, depth-limited. |
| Tree ML | **LightGBM** | same | gradient-boosted trees, early-stopped. |
| Tree ML | CatBoost | same | ordered boosting, early-stopped. |
| Deep | GRU | 30 × 28–29 window | 1 GRU layer (32 units) + dropout + dense. |
| Deep | LSTM | same | same shape as GRU. |
| Ensemble | Stacked | OOF predictions | non-negative Ridge over base models (experiment). |

**Two-phase training.** Models with a stopping rule (trees, boosting iterations, epochs) are fitted on train while monitoring validation to choose the stopping point (phase A, gives honest validation metrics and loss curves), then refitted on train+val with that stopping point fixed (phase B, the deployed model). Validation is never inside its own early-stopping monitor.

## 6. Validation and model selection

**Expanding-window walk-forward validation** (`src/evaluation/cross_validation.py`), 4 folds over the train+val period:

```
fold 1: train [0 … 60 %)          validate next block
fold 2: train [0 … 70 %)          validate next block
fold 3: train [0 … 80 %)          validate next block
fold 4: train [0 … 90 %)          validate last block
```

Inside each fold the last 15 % of the fold's training window is the early-stopping monitor. Hyper-parameters (`src/training/tune_models.py`) and the served model per asset (`src/evaluation/backtesting.py`) are chosen by **mean fold RMSE of the return** — the test set is never loaded by either script.

**Why walk-forward rather than k-fold.** k-fold would train on 2026 data to predict 2024, which is impossible in deployment and leaks regime information. Walk-forward always predicts forward in time and yields several out-of-sample estimates, so we can report mean ± std instead of one lucky window.

## 7. Evaluation (`src/utils/metrics.py`)

| Metric | Space | Why |
|---|---|---|
| MAE, RMSE | USD | What a user cares about; RMSE penalises large misses. |
| MAPE | USD | Scale-free across assets ($60 k vs $60). |
| RMSE, MAE, **R²** | return | The quantity actually predicted. R² < 0 means worse than predicting the mean. R² on **price** is not reported (random walk ≈ 0.95). |
| **Directional accuracy** | sign | % of non-flat days where sign(ŷ) = sign(y), with a one-sided binomial p-value against 50 %. Days with exactly zero return are excluded (their direction is undefined), and a zero forecast has no direction. |
| **Diebold–Mariano** | return | Formal test of H0 "equal squared-error accuracy" vs the naive forecast (HAC variance, Harvey correction). |
| Strategy backtest | — | Long/flat rule, 10 bps cost, vs buy-and-hold: economic relevance. |
| Train vs validation RMSE | return | Over-fitting gap (`results/train_val_metrics.csv`, `figures/overfitting_gap.png`). |

## 7b. Design experiments (`src/experiments/run_experiments.py`)
E1 data size (3-year vs full history on the fixed validation window), E2 feature-group ablation, E3 horizon (1-day vs 5-day return), E4 22-day realised-volatility target — all on validation folds only, logged to `results/experiments/` with a Markdown summary. `src/experiments/before_after.py` compares the archived 3-year system with the current one on identical unseen days.

## 8. Reproducibility

`src/utils/reproducibility.py::set_all_seeds(42)` seeds Python, NumPy and TensorFlow; tree models use `random_state=42`. Pinned `requirements.txt`. `make pipeline` (or `python scripts/retrain.py --no-fetch`) reproduces every number in `results/` from `data/raw/`.

## 9. Leakage checklist (what was verified)

| Check | Result | Evidence |
|---|---|---|
| Random shuffling | none | chronological slicing by date in `normalize_and_split` |
| Scaler fitted on test | no | `tests/test_sequences_and_split.py::test_scaler_fitted_on_train_only` |
| Target inside the feature window | no | `test_create_sequences_target_is_next_row_and_never_in_window` |
| Rolling/EWM/shift look-ahead | none | `test_no_lookahead_in_any_feature` |
| External data alignment | same-day for BTC (bar closes after the US close); previous day for the metals (13:30 ET settlement precedes the macro closes) | `_align_external`; `test_commodity_external_features_are_previous_day_values` |
| Futures High/Low past the settlement | `hl_range`, `atr_norm`, `ADX` lagged one session for metals | `lag_post_settlement_features`; `test_commodity_high_low_features_are_lagged_one_session` |
| Early stopping on training data | fixed | registry two-phase fit |
| Hyper-parameters chosen on test | no | tuner never calls `build_dataset()[...test]` |
| Synthetic rows | none | `test_commodities_keep_trading_calendar` |
| Price reconstruction | exact | `test_true_target_reconstructs_actual_close_exactly` |
| Test set touched more than once | no | only `backtesting.py` reads `X_test` |
| Multi-day targets straddling a split | no | split by target dates; `test_task_targets_are_future_only_and_embargoed` |
| Demo prediction using future rows | no | `predict_for_date` slices `features.loc[:as_of]`; `test_predict_for_date_uses_only_past_data_and_reveals_actual` perturbs every row after the chosen day and asserts the forecast is unchanged |
