# FYP Phase 2 — Final Audit, Prediction Assessment and Viva Preparation

*AI-Powered Commodity Price Predictor (Bitcoin · Gold · Silver) — audit performed 2026-10-04 on commit `afafa9d` + the changes listed in §24.*
*Every number below was read from the project's own result files or re-computed by its code during this audit. Nothing is estimated.*

---

## 0. The answer first

| Question | Answer (evidence) |
|---|---|
| Are the predictions good? | **They are honest, but they have no skill.** On the unseen test window and on ~6 years of pooled walk-forward predictions, no model and no ensemble beats "tomorrow = today" (random walk). |
| Is the project defensible? | **Yes.** The methodology is clean: no leakage was found, it reproduces bit-for-bit from raw data, validation is time-ordered, it uses proper significance tests and reports its own failure. A rigorous negative result is a legitimate FYP outcome. |
| What did this audit change? | It added a long-run (pooled walk-forward) significance test, calibrated the 68 % uncertainty band, hardened inference against feature-order errors, and tested 3 more improvement ideas (all rejected on validation). **Point forecasts are unchanged.** |
| Final status | 🟡 **READY AFTER FINAL MINOR FIXES**: commit the work and correct the uncertainty-band paragraph in the PDF report (§23). The software and the ML methodology are final. |

---

## PHASE 1 — Complete inspection (what exists, verified in code)

| Area | Files | What it does (verified) |
|---|---|---|
| Config | `config.py` | Paths, assets (BTC-USD, GC=F, SI=F), `DATA_START_DATE=2018-01-01`, frozen split `TRAIN_END=2025-09-10`, `VAL_END=2026-02-17`, target `log_return`, horizon 1, feature policy (`LEVEL_COLUMNS` never used as inputs), close-time rule (`EXTERNAL_SAME_DAY`, `POST_SETTLEMENT_FEATURES`), `SEQ_LEN=30`, default hyper-parameters, `CV_FOLDS=4` |
| Data download | `src/data/data_collection.py`, `external_data.py`, `market_calendar.py` | Yahoo Finance OHLCV + DXY, WTI, VIX, 10-y yield, S&P 500, alternative.me Fear & Greed; **complete-bar rule** drops the still-running bar |
| Preprocessing | `src/data/preprocessing.py` | Dedup/sort, BTC calendar fill, metals keep exchange days, log return, 28–29 stationary features, close-time alignment, `dropna`, validation (NaN/inf/row count), chronological split, **MinMaxScaler fitted on train only**, `build_dataset()` = the single loader for every stage |
| Live data | `src/data/sync_live_data.py` | Downloads into `data/raw_live/`, writes `*_live_features.csv` atomically; frozen data never touched |
| Models | `src/models/registry.py`, `model_gru.py`, `model_lstm.py`, `ensemble_model.py` | Naive (random walk), Naive-Mean (drift), ARIMA(p,0,q) by AIC, Ridge, RandomForest, LightGBM, CatBoost, GRU, LSTM; two-phase fitting; stacked ensemble with non-negative Ridge (experiment) |
| Tuning | `src/training/tune_models.py` | Grid search scored by 4-fold expanding-window walk-forward CV on train+val only → `results/tuning/best_params.json` |
| Training | `src/training/train_models.py` | Phase A: fit on train, monitor val (metrics, stopping point); Phase B: refit on train+val with that stopping point → deployed model |
| Evaluation | `src/evaluation/cross_validation.py`, `backtesting.py`, `plots.py` | Walk-forward CV table, single test evaluation of every model + Combined, pooled walk-forward tests (new), band calibration (changed), regime analysis, `model_status.json`, `FINAL_RESULTS.md` |
| Metrics | `src/utils/metrics.py` | RMSE/MAE/R² (return), MAE/RMSE/MAPE (USD), directional accuracy + binomial p, UP-call share, Diebold–Mariano (HLN-corrected), long/flat backtest with 10 bps, band coverage |
| Inference | `src/inference/prediction.py` | `predict_next_day` (live), `predict_for_date` (frozen, demo), Combined = equal-weight mean of 6 models, 68 % band |
| UI / API | `app/streamlit_app.py`, `src/api/app.py` | 4-tab dashboard; FastAPI `/health`, `/models`, `/predict/{asset}` |
| Experiments | `src/experiments/run_experiments.py`, `before_after.py`, `audit_improvements.py` (new) | E1 data size, E2 feature-group ablation, E3 horizon, E4 volatility target, E5 audit candidates |
| Tests / CI | `tests/` (45 tests), `.github/workflows/ci.yml` | Look-ahead, close-time alignment, complete-bar rule, target alignment, split, reconstruction, metrics, inference, band, failure modes |
| Saved models | `data/models/` (git-ignored except `model_status.json`) | 18 trained models + 3 stacks + 3 scalers |

Nothing claimed in the README is missing from the code. Nothing important exists in the code without being documented.

---

## PHASE 2 — Project health check (state found at the start of the audit)

| Area | Score /10 | Status | Problems found |
|---|--:|---|---|
| Dataset | 7 | 🟡 | Free Yahoo daily data (2018-01 → 2026-09). Continuous futures can be revised after contract rolls (seen: up to 1.4 % on Gold's live series). The test window is short (143–207 days). |
| Data quality | 8 | 🟢 | Complete-bar rule, dedup, positive prices, NaN/inf validation; no synthetic weekend rows for futures |
| Preprocessing | 9 | 🟢 | Correct and reproducible; the scaler is fitted on train only |
| Target | 9 | 🟢 | Next-day log return, stationary; the random-walk baseline is explicit |
| Features | 6.5 | 🟡 | Sensible and backward-looking, but heavily redundant (15–18 pairs with \|ρ\| > 0.85) and without out-of-sample signal |
| Leakage prevention | 9.5 | 🟢 | Close-time rule, target never in the window, train-only scaler, tests; no leak found (§6) |
| Time-series validation | 9 | 🟢 | Chronological split and expanding walk-forward with embargo. The test window *was inspected during development*. |
| Models | 7 | 🟡 | Broad and appropriate. GRU/LSTM early-stop after 1–9 epochs, so they are effectively constant predictors; for BTC they are significantly worse than the random walk. |
| Hyper-parameters | 8 | 🟢 | Walk-forward grid on train+val only. LightGBM's `subsample=0.8` is inert (no `subsample_freq`). |
| **Prediction quality** | **3** | 🔴 | No model beats the random walk (DM p ≥ 0.17 on test). Metals' "direction" is drift: they say UP on 91–97 % of test days. This is a result, not a bug. |
| Ensemble | 7 | 🟡 | Equal weights, nothing selected on test; no better than the best single model and no worse |
| Evaluation | 9 | 🟢 | DM, binomial, UP calls, strategy, regimes. It lacked a long-run significance test (added). |
| Uncertainty band | 6 | 🟡 | Labelled "68 %" but covered 74–75 % of test days (fat tails): mis-calibrated (fixed) |
| Streamlit | 8.5 | 🟢 | All tabs work, honest captions, live refresh with vendor-lag warning |
| Code quality | 8 | 🟢 | Modular, single loader, explicit failure modes. Inference depended on the column order of the feature file (hardened). |
| Reproducibility | 9.5 | 🟢 | Seeds and pinned versions; the full re-run reproduces every result file to ≤ 1e-13 |
| Documentation | 7 | 🟡 | README accurate. The PDF report's band paragraph was already out of date ("±1 RMSE"); test count and band text stale. |
| Presentation readiness | 8.5 | 🟢 | Demo flow works end to end (§21) |
| **Final FYP readiness** | **8.5** | 🟡 | |

**Initial verdict: 🟡 FINAL-READY AFTER MINOR FIXES.** None of the issues is a methodological flaw. The weak "prediction quality" score is the scientific finding, and it must be *presented* correctly rather than "fixed".

---

## PHASE 3 — What exactly is predicted

| Item | Value |
|---|---|
| Assets | Bitcoin (`BTC-USD`, spot, 7 days/week), Gold (`GC=F`, COMEX front-month futures), Silver (`SI=F`) |
| Frequency | Daily bars |
| Horizon | 1 trading day (BTC: next calendar day; metals: next exchange day) |
| Target | **Next-day log return** (not the price) |
| Output to user | Price, reconstructed from the predicted return |
| Inputs | 28 (Gold), 29 (BTC, Silver) stationary features of day *t* (tabular models), or the last 30 days of them (GRU/LSTM) |

**Mathematical definition** (`preprocessing._task_target`, `build_dataset`):

```
P_t        = close of day t (BTC: 00:00 UTC bar close; GC=F/SI=F: 13:25–13:30 ET COMEX settlement)
y_t        = ln(P_{t+1} / P_t)                         ← the target, indexed by the day t the forecast is made
z_t        = (y_t − μ_train) / σ_train                 ← what the models are trained on (μ, σ from the TRAIN split only)
ŷ_t        = ẑ_t · σ_train + μ_train                   ← de-standardised prediction
P̂_{t+1}    = P_t · exp(ŷ_t)                            ← price shown to the user
Combined:  ŷ_t = (1/6) Σ_m ŷ_t^(m),  m ∈ {Ridge, RandomForest, LightGBM, CatBoost, GRU, LSTM}
Band:      [P_t · exp(ŷ_t − k·σ_t),  P_t · exp(ŷ_t + k·σ_t)],  σ_t = EWMA volatility at t,  k calibrated (§11)
Random walk (Naive): ŷ_t = 0  ⇔  P̂_{t+1} = P_t
```

Splits are assigned by the date the *target* covers (t+1). So a training target never reaches past `TRAIN_END`, and a validation target never reaches past `VAL_END`.

---

## PHASE 4 — Real prediction performance

Split sizes (`model_status.json`): **Bitcoin** train 2,750 · validation 160 · test 207 (test targets 2026-02-18 → 2026-09-12). **Gold / Silver** train 1,874 · validation 109 · test 143 (2026-02-18 → 2026-09-11).

### 4.1 Untouched TEST set (evaluated once; `results/final_test_results.csv`)

MAE/RMSE/MAPE in USD on the reconstructed price; RMSE and R² on the return; DA = direction correct on non-flat days (one-sided binomial p vs 50 %); DM p = Diebold–Mariano vs the random walk.

**Bitcoin** (49 % of test days were up days)

| Model | MAE ($) | RMSE ($) | MAPE | RMSE (ret) | R² (ret) | DA % (p) | UP calls | DM p |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Stacked (experiment) | 1,061.35 | 1,442.94 | 1.52 % | 0.02065 | +0.013 | 50.2 (0.50) | 64 % | 0.44 |
| LightGBM | 1,076.49 | 1,446.18 | 1.54 % | 0.02068 | +0.010 | 51.2 (0.39) | 45 % | 0.79 |
| **Combined (served)** | 1,063.94 | 1,447.37 | 1.52 % | 0.02072 | +0.006 | 46.9 (0.83) | 37 % | 0.49 |
| RandomForest | 1,064.48 | 1,448.81 | 1.52 % | 0.02075 | +0.004 | 48.3 (0.71) | 57 % | 0.62 |
| Naive-Mean (drift) | 1,067.32 | 1,451.38 | 1.53 % | 0.02078 | −0.000 | 49.3 (0.61) | 100 % | 0.81 |
| **Naive (random walk)** | 1,066.42 | 1,451.73 | 1.52 % | 0.02079 | −0.001 | — | — | — |
| Ridge | 1,071.97 | 1,452.59 | 1.53 % | 0.02080 | −0.002 | 47.8 (0.76) | 39 % | 0.91 |
| LSTM | 1,068.06 | 1,453.44 | 1.53 % | 0.02082 | −0.004 | 46.4 (0.87) | 26 % | 0.35 |
| CatBoost | 1,071.07 | 1,453.76 | 1.53 % | 0.02082 | −0.004 | 46.9 (0.83) | 60 % | 0.74 |
| GRU | 1,075.86 | 1,458.31 | 1.54 % | 0.02089 | −0.010 | 46.9 (0.83) | 16 % | 0.35 |
| ARIMA | 1,067.96 | 1,459.44 | 1.53 % | 0.02091 | −0.012 | 46.9 (0.83) | 72 % | 0.35 |

**Gold** (50 % up days)

| Model | MAE ($) | RMSE ($) | MAPE | RMSE (ret) | R² (ret) | DA % (p) | UP calls | DM p |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| RandomForest | 58.08 | 75.10 | 1.29 % | 0.01659 | +0.012 | 49.0 (0.63) | 87 % | 0.57 |
| GRU | 58.00 | 75.42 | 1.29 % | 0.01664 | +0.007 | 51.7 (0.37) | 97 % | 0.58 |
| **Combined (served)** | 58.20 | 75.73 | 1.29 % | 0.01671 | −0.001 | 50.3 (0.50) | 97 % | 0.97 |
| **Naive (random walk)** | 58.26 | 75.76 | 1.29 % | 0.01671 | −0.002 | — | — | — |
| LightGBM | 58.25 | 75.81 | 1.29 % | 0.01672 | −0.003 | 49.7 (0.57) | 100 % | 0.88 |
| Naive-Mean (drift) | 58.39 | 75.99 | 1.30 % | 0.01675 | −0.007 | 49.7 (0.57) | 100 % | 0.43 |
| Ridge | 58.51 | 76.05 | 1.30 % | 0.01677 | −0.009 | 50.3 (0.50) | 94 % | 0.76 |
| CatBoost | 58.43 | 76.22 | 1.30 % | 0.01681 | −0.015 | 51.7 (0.37) | 92 % | 0.71 |
| ARIMA | 58.57 | 76.27 | 1.30 % | 0.01681 | −0.015 | 50.3 (0.50) | 80 % | 0.52 |
| LSTM | 58.82 | 76.58 | 1.31 % | 0.01687 | −0.022 | 49.7 (0.57) | 100 % | 0.28 |
| Stacked (experiment) | 58.95 | 77.28 | 1.31 % | 0.01705 | −0.043 | 51.0 (0.43) | 97 % | 0.44 |

**Silver** (52 % up days)

| Model | MAE ($) | RMSE ($) | MAPE | RMSE (ret) | R² (ret) | DA % (p) | UP calls | DM p |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| RandomForest | 1.76 | 2.33 | 2.44 % | 0.03146 | +0.026 | 56.6 (0.07) | 66 % | 0.17 |
| **Combined (served)** | 1.78 | 2.37 | 2.47 % | 0.03180 | +0.005 | 53.8 (0.20) | 91 % | 0.46 |
| Ridge | 1.79 | 2.37 | 2.48 % | 0.03181 | +0.005 | 54.5 (0.16) | 94 % | 0.48 |
| LightGBM | 1.79 | 2.37 | 2.48 % | 0.03189 | −0.001 | 58.7 (0.02) | 86 % | 0.98 |
| GRU | 1.79 | 2.37 | 2.48 % | 0.03189 | −0.001 | 55.2 (0.12) | 88 % | 0.99 |
| **Naive (random walk)** | 1.79 | 2.37 | 2.49 % | 0.03189 | −0.001 | — | — | — |
| Stacked (experiment) | 1.79 | 2.38 | 2.48 % | 0.03191 | −0.002 | 52.4 (0.31) | 94 % | 0.91 |
| Naive-Mean (drift) | 1.79 | 2.38 | 2.49 % | 0.03192 | −0.003 | 51.7 (0.37) | 100 % | 0.63 |
| CatBoost | 1.79 | 2.38 | 2.49 % | 0.03193 | −0.003 | 51.7 (0.37) | 100 % | 0.70 |
| LSTM | 1.80 | 2.38 | 2.49 % | 0.03198 | −0.006 | 51.7 (0.37) | 100 % | 0.63 |
| ARIMA | 1.81 | 2.37 | 2.52 % | 0.03226 | −0.024 | 52.4 (0.31) | 55 % | 0.50 |

R² on *price levels* is deliberately not reported: the random walk alone scores ≈ 0.95 there, so it says nothing about skill.

### 4.2 TRAINING vs VALIDATION (phase A, `results/train_val_metrics.csv`), return RMSE

| Asset | Model | Train RMSE | Val RMSE | Train R² | Val R² | Train DA | Val DA | Stopping point |
|---|---|--:|--:|--:|--:|--:|--:|---|
| Bitcoin | Ridge | 0.0335 | 0.0261 | +0.005 | −0.012 | 52.3 | 50.6 | — |
| | RandomForest | 0.0330 | 0.0261 | +0.036 | −0.016 | 56.0 | 51.3 | — |
| | LightGBM | 0.0274 | 0.0254 | **+0.335** | +0.038 | **71.5** | 54.4 | 81 trees |
| | CatBoost | 0.0321 | 0.0260 | +0.088 | −0.011 | 58.5 | 44.4 | 50 iters |
| | GRU | 0.0336 | 0.0260 | −0.001 | −0.006 | 50.0 | 53.8 | 1 epoch |
| | LSTM | 0.0336 | 0.0262 | +0.001 | −0.021 | 50.4 | 46.3 | 2 epochs |
| Gold | Ridge | 0.0097 | 0.0202 | +0.006 | −0.008 | 54.2 | 56.9 | — |
| | RandomForest | 0.0095 | 0.0203 | +0.047 | −0.014 | 57.8 | 54.1 | — |
| | LightGBM | 0.0097 | 0.0203 | +0.005 | −0.011 | 54.3 | 63.3 | 10 trees |
| | CatBoost | 0.0093 | 0.0200 | +0.088 | +0.021 | 57.4 | 59.6 | 606 iters |
| | GRU | 0.0097 | 0.0202 | +0.004 | −0.006 | 54.3 | 63.3 | 8 epochs |
| | LSTM | 0.0098 | 0.0203 | +0.000 | −0.011 | 54.3 | 63.3 | 1 epoch |
| Silver | Ridge | 0.0192 | 0.0554 | +0.005 | −0.012 | 52.9 | 59.6 | — |
| | RandomForest | 0.0187 | 0.0554 | +0.055 | −0.011 | 60.2 | 59.6 | — |
| | LightGBM | 0.0192 | 0.0553 | +0.001 | −0.008 | 52.4 | 62.4 | 10 trees |
| | CatBoost | 0.0190 | 0.0553 | +0.017 | −0.006 | 53.4 | 64.2 | 82 iters |
| | GRU | 0.0192 | 0.0553 | +0.003 | −0.006 | 51.3 | 63.3 | 9 epochs |
| | LSTM | 0.0192 | 0.0554 | +0.001 | −0.012 | 48.9 | 56.0 | 4 epochs |

Random-walk validation RMSE: BTC 0.02610, Gold 0.02035, Silver 0.05536. These validation numbers are *optimistic*, because the same window chose the stopping point. They are diagnostics, not final performance. The metals' validation window (Sept 2025 → Feb 2026) was a volatility shock (Silver's daily RMSE roughly tripled), and validation DA of 59–64 % equals the share of up days in a rallying market.

### 4.3 Walk-forward VALIDATION (4 expanding folds inside train+val; `results/cv_results.csv`), return RMSE mean

| Model | Bitcoin | Gold | Silver |
|---|--:|--:|--:|
| Naive | 0.03160 | 0.01114 | 0.02391 |
| Combined | 0.03159 | 0.01113 | 0.02390 |
| best single (by CV) | CatBoost 0.03156 | CatBoost 0.01110 | GRU 0.02387 |

---

## PHASE 5 — Baseline test

The project **does** have baselines built in, and every model is compared against them on identical days: **Naive** (random walk, ŷ = 0, i.e. P̂_{t+1} = P_t), **Naive-Mean** (always predict the training-window mean return, the "drift" or "always UP" forecast) and **ARIMA** (AIC-selected order, walk-forward). No baseline was added after the fact.

| Test-set RMSE (return) | Bitcoin | Gold | Silver |
|---|--:|--:|--:|
| Random walk | 0.02079 | 0.01671 | 0.03189 |
| Best single model on test | Stacked 0.02065 (−0.7 %) | RandomForest 0.01659 (−0.7 %) | RandomForest 0.03146 (−1.4 %) |
| Combined (served) | 0.02072 (−0.3 %) | 0.01671 (−0.04 %) | 0.03180 (−0.3 %) |
| DM p, Combined vs random walk | 0.49 | 0.97 | 0.46 |

**Does the ML system add predictive value beyond the baseline? No.** Every difference is under 1.5 % and none is statistically significant on test (the smallest DM p is 0.17). The same holds on the 6-year pooled record (§7). "Best on test" picks are hindsight: the model that is best on test was not the best on validation for any asset.

---

## PHASE 6 — Data-leakage audit (strict, traced in code)

| Check | Where | Verdict |
|---|---|---|
| Target leakage | `create_sequences` / `build_dataset`: window = rows t−29…t; target = return t→t+1 (`_task_target` uses `shift(-k)` only for the target) | ✅ none. Tested (`test_sequences_and_split.py`). |
| Future information in features | All indicators use `rolling`, `ewm`, `shift(+k)`, or `ta` (backward). A truncation test from the 2026-09-21 audit rebuilt features from history cut at test dates: unchanged. | ✅ none |
| Close-time (vendor) leakage | Metals settle 13:30 ET, before the US macro closes → macro values from the latest date *strictly before* t (`_align_external`); High/Low features lagged one session. BTC closes 00:00 UTC, after the US close → same-day macro allowed. | ✅ fixed 2026-09-13 (the cause of an old fake "62 % Silver"). Tested. |
| Silver's same-day `gold_return` | Gold settles 13:30, Silver 13:25: 5 minutes of gold trading fall inside Silver's target interval | ⚠️ *negligible*: the 5-minute slice is ≈ 0.36 % of a day's variance, so its correlation with the next-day target is bounded at ≈ 0.003 (R² ≈ 1e-5). Documented. |
| Rolling-window leakage | No centred windows; `return_20d` = sum of the last 20 returns *including* t (known at t) | ✅ |
| Scaling leakage | `MinMaxScaler().fit(train_df)` only; val/test transformed | ✅ |
| Target standardisation | μ, σ from `y_real_train` only | ✅ |
| Preprocessing leakage | `dropna` and the BTC forward-fill run before the split, but they only use past rows | ✅ |
| Train/test contamination | Masks by target date; ARIMA appends truth only *after* forecasting each day; inference uses `feats.loc[:as_of]` (tested by perturbing every future row) | ✅ |
| Hyper-parameter leakage | Tuning uses train+val folds only; the test split is read only in `backtesting.py` | ✅ (but the val metrics are optimistic — see §4.2) |
| Ensemble-weight leakage | Combined has no weights. The stack is fitted on walk-forward OOF of train+val. | ✅ |
| Random shuffling | Keras `shuffle=False`; `TimeSeriesSplit`; no `train_test_split` anywhere | ✅ |
| Future timestamps / partial bars | Complete-bar rule drops the running bar (tested) | ✅ |
| **Test-set reuse** (methodological) | The test window was looked at in several development rounds (leak fix, Combined, band) | ⚠️ disclosed. No *parameter* was fitted on test, but it is not a pristine one-shot holdout, which is why the pooled walk-forward evidence (§7) matters. |

**No leakage requiring a fix was found**, so no before/after retrain is needed. The full pipeline was re-run from raw data: every committed result file reproduced to ≤ 1e-13 and the scaled splits are byte-identical.

---

## PHASE 7 — Time-series validation

| Element | Implementation |
|---|---|
| Chronological split | Frozen dates: train ≤ 2025-09-10 < validation ≤ 2026-02-17 < test |
| Validation set | Phase-A stopping point (trees/iterations/epochs) |
| Walk-forward | `TimeSeriesSplit(4)` on train+val: an expanding training window and 4 consecutive validation blocks; an inner 15 % tail of each fold's training window acts as the early-stopping monitor |
| Embargo | For multi-day targets (experiments) the last h−1 training samples are purged; the split is by target date |
| Test | Read only by `backtesting.py` |
| **New: pooled walk-forward** | All 4 folds' out-of-fold predictions pooled: **2,328 days (BTC) / 1,584 days (metals), Oct 2019 → Feb 2026**, then DM vs random walk, DM vs drift and the binomial direction test (`results/walkforward_pooled.csv`) |

**Pooled walk-forward results (train+val only):**

| Asset | Combined RMSE vs RW | DM p | DA % (p vs 50 %) | UP calls / up days | Drift forecast DA | Best single |
|---|--:|--:|--:|--:|--:|---|
| Bitcoin | −0.018 % | 0.87 | 50.8 (0.23) | 67 % / 51 % | 48.8 % | CatBoost −0.09 % (p 0.45). GRU/LSTM **worse** than RW (p 0.04 / 0.004). |
| Gold | −0.213 % | 0.09 | 52.5 (0.02) | 74 % / 56 % | **55.8 %** | CatBoost −0.44 % (p 0.28) |
| Silver | −0.058 % | 0.69 | 52.7 (0.02) | 78 % / 53 % | **53.5 %** | GRU −0.17 % (p 0.42) |

The metals' "significant" direction hit-rates are *below* what "always UP" achieves on the same days: drift, not timing. **Is the final evaluation genuinely unseen, future-like data?** Yes for parameters: the test window is strictly after every training and validation target, and no weight or stopping point was fitted on it. It is not pristine as a design-decision holdout (disclosed above). Its conclusion matches the 10× larger pooled record, and a forward check on 15–20 days that *no* decision had seen (2026-09-13 → 10-02) agrees: RMSE within ±3.5 % of RW, DA 40–47 %. Validation is trustworthy.

---

## PHASE 8 — Overfitting

- **LightGBM (BTC)**: train R² 0.335 and train DA 71.5 %, vs validation R² ≈ 0.04 and DA 54 %. That is memorisation, which is why early stopping cut it to 81 trees and the walk-forward grid picked heavy regularisation elsewhere. Its test R² is +0.010 (DM p 0.79): no harm, no skill.
- **CatBoost / RandomForest**: train R² 0.04–0.09 vs validation ≤ 0.02. A mild gap, controlled by depth 3–7 and min-leaf 30.
- **GRU / LSTM**: stop at **1–9 epochs**. They do not overfit; there is nothing to learn, so validation loss is minimal almost immediately and they output ≈ constant returns (prediction std 0.02–0.2 %).
- **LightGBM (metals)**: 10 trees (the floor), i.e. ≈ a constant.
- **Validation vs test**: the test RMSE ranking is uncorrelated with the validation ranking. That is a sign of noise-level differences, not overfitting to validation.

**Cause:** a signal-to-noise ratio near zero (univariate feature–target correlations ≤ 0.065 on train, and they do not persist out-of-sample). **Solution already in place:** strong regularisation chosen by walk-forward CV, early stopping, single-layer RNNs, and forecast averaging. More regularisation would only push every model further towards the random walk, which it already equals. No further change is justified.

---

## PHASE 9 — Feature-engineering audit

All features are computed from data up to the close of day *t* (metals: previous-session values where marked †). Price-level columns (open/high/low/price/volume/EMA/BB/ATR/MACD) are **chart-only** and never model inputs.

| Feature | Formula | Source | Assets | Leakage risk | Usefulness (evidence) | Redundant with | Decision |
|---|---|---|---|---|---|---|---|
| `log_return` | ln(P_t/P_{t−1}) | own close | all | none | small negative autocorr (BTC ρ = −0.05 train) | — | keep |
| `return_1d, _2d, _5d, _10d` | r_{t−k} (lagged returns) | own | all | none | top CatBoost inputs (BTC); no OOS signal | — | keep |
| `return_20d` | Σ_{i=0..19} r_{t−i} | own | all | none | top Gold CatBoost feature (0.21) | ema30_ratio (0.91), macd_norm (0.88) | keep |
| `RSI` | Wilder RSI(14) | own | all | none | low | ema30_ratio (0.93–0.95) | keep |
| `ADX` † | ADX(14) from H/L/C | own | all | metals lagged | low | — | keep |
| `ROC` | 12-day % change | own | all | none | low | ema14_ratio (0.86–0.89) | tested removal: no gain |
| `ema14_ratio, ema30_ratio` | P_t/EMA−1 | own | all | none | Silver's top feature (0.44) | RSI, return_20d | tested removal: worse |
| `macd_norm, macd_hist_norm` | MACD/P, (MACD−signal)/P | own | all | none | low | return_20d | keep |
| `bb_pctb, bb_width` | position in / width of 20-day ±2σ bands | own | all | none | low | RSI / vol | keep |
| `atr_norm` †, `hl_range` † | ATR(14)/P, (H−L)/P | own H/L | all | metals lagged (H/L span past settlement) | volatility state | ewma_vol (0.87–0.94) | keep |
| `volatility_10d/30d` | rolling std of r | own | all | none | volatility state | ewma_vol (0.87–0.96) | tested removal: worse |
| `rv_1d, rv_5d, rv_22d` | HAR realised vol | own | all | none | volatility state; E4 shows vol is forecastable | ewma_vol (0.95–0.96) | keep |
| `ewma_vol` | RiskMetrics λ = 0.94 | own | all | none | **drives the served band** | — | keep |
| `dow_sin, dow_cos` | day-of-week cycle | calendar | BTC | none | 2nd BTC CatBoost feature, no OOS value | — | keep |
| `log_volume_change` | ln(V_t/V_{t−1}) clipped | own | BTC | none | low | — | keep (futures volume unusable) |
| `sp500_return, vix_return` | log change | Yahoo | all | BTC same-day (closes after US); metals previous day | low | — | keep |
| `dxy_return, oil_return, tnx_return` | log change | Yahoo | metals | previous day | Gold CatBoost #2/#3 | — | keep |
| `fear_greed` | index 0–100 | alternative.me | BTC | published 00:00 UTC before the bar closes | low | — | keep |
| `gold_return` | gold's same-day log return | Yahoo | Silver | 5-min settlement gap (§6), negligible | Silver #3 | — | keep |

**Keep:** all of them (removing the 10 redundant ones was tested in E5b and made every tabular model equal or worse). **Remove:** none justified by evidence. **Worth testing:** done (E2 group ablation, E5b de-duplication). **Not added:** more technical indicators. E2 shows that no feature *group* moves walk-forward RMSE by more than 0.4 %, and adding inputs without signal only adds variance.

---

## PHASE 10 — Candidate improvements

| Improvement | Why | Expected benefit | Complexity | Leakage risk | Worth it? |
|---|---|---|---|---|---|
| Long-run pooled walk-forward significance | One 143–207-day test has little power | Credibility of the conclusion | Low | none (train+val only) | ✅ **implemented** |
| Calibrate the 68 % band | Band over-covers 73–75 % in all 12 training folds | A correctly labelled interval | Low | low (fit on training errors only) | ✅ **implemented** |
| Feature-order guard at inference | Silent wrong predictions if a file's column order changes | Robustness | Low | none | ✅ **implemented** |
| Drop redundant features | 15–18 pairs \|ρ\| > 0.85 | Less variance | Low | none | tested (E5b) → ❌ |
| Ensemble membership (tabular only) | BTC RNNs are worse than RW | Small | Low | none | tested (E5a) → ❌ |
| Zero-drift forecast | Metals' UP bias = training drift | Removes the drift bias | Low | none | tested (E5c) → ❌ |
| More lags / rolling stats / indicators | — | ≈ 0 (E2: no group matters) | Low | low | ❌ not justified |
| XGBoost | Same family as LightGBM/CatBoost | ≈ 0 | Low | none | ❌ redundant |
| ExtraTrees, HistGB, validation-weighted ensembles | Tried 2026-09-21: ±0.2 % of RW | ≈ 0 | Low | none | ❌ (already tested) |
| Bigger LSTM/GRU, Transformers | Stop at 1–9 epochs: there is no pattern to fit | negative (more variance) | High | none | ❌ |
| Stacking | Exists as an experiment: worst model on Gold test | ≈ 0 / negative | — | none | kept as experiment, not served |
| Longer history (pre-2018) | E1: 3 y vs 8 y gave mixed signs | ≈ 0 | Medium | none | ❌ |
| More external variables | E2 macro group ≈ 0; each new series needs a close-time check | ≈ 0, adds leak risk | Medium | **high** | ❌ |
| Make volatility the served target | E4: Ridge beats persistence by 11.5 % (BTC) | real skill | High (changes the project question) | low | ❌ at the final stage; presented as a finding and future direction |

---

## PHASE 11 — Improvements actually tested

Decision rule fixed **before** looking at results: keep a point-forecast change only if it lowers the pooled walk-forward RMSE **for every asset** with DM p < 0.05 against the served forecast. The test set was read only after the decision, once.

| # | Old methodology | New | Validation result (pooled walk-forward) | Test result | Kept? |
|---|---|---|---|---|---|
| I1 | Significance only on 143–207 test days + CV means | Pooled walk-forward DM/binomial on 1,584–2,328 days | — (evaluation, not a model) | — | ✅ yes: strengthens the conclusion |
| I3 | Band ±1·σ_t, labelled 68 % | Band ±k·σ_t, k = 68.27 % quantile of \|e\|/σ_t on *training-split* OOF errors: k = 0.825 / 0.909 / 0.870 | Training folds: ±1σ covered 73–76 % in 12/12 folds. Validation (out-of-sample): 60.6 / 66.1 / 54.1 % vs 75.0 / 72.5 / 57.8 % → better for Gold, worse for BTC and Silver. | **65.7 / 69.2 / 69.2 %** vs 73.9 / 75.5 / 75.5 % (target 68.3) → better for all three | ✅ yes (structural fat-tail effect, consistent in the large training sample and on test). Validation mixed; disclosed. |
| E5a | Combined of 6 | Combined of 4 tabular | BTC −0.09 % (p 0.04), Gold −0.04 % (p 0.50), Silver +0.03 % (p 0.49) | not used | ❌ fails "every asset" |
| E5b | 28–29 features | 18–19 (redundant removed) | BTC −0.00 %, Gold +0.02 %, Silver +0.15 % vs served; each tabular model equal or worse | not used | ❌ |
| E5c | Combined | Combined − training mean (zero drift) | BTC −0.13 % (p 0.03), Gold **+0.17 % worse** (p 0.03), Silver +0.03 % | not used | ❌. It would probably have looked better on the falling 2026 metals test window; adopting it for that reason would be test-set snooping. |

Files: `results/walkforward_pooled.csv`, `results/band_calibration.csv`, `results/experiments/E5_audit_candidates.csv`, `E5_summary.md`.

---

## PHASE 12 — Final model comparison

Primary metric: **return RMSE relative to the random walk** (what the models predict), with the DM p-value. Test window, and in brackets the pooled walk-forward record:

| Model | Bitcoin | Gold | Silver |
|---|--:|--:|--:|
| Baseline: random walk | 0.02079 (0.0323) | 0.01671 (0.0114) | 0.03189 (0.0247) |
| Ridge | +0.04 % [+0.05 %] | +0.33 % [−0.21 %] | −0.27 % [+0.05 %] |
| RandomForest | −0.23 % [−0.00 %] | −0.73 % [+0.07 %] | −1.36 % [+0.12 %] |
| LightGBM | −0.53 % [−0.03 %] | +0.05 % [−0.15 %] | −0.01 % [−0.07 %] |
| CatBoost | +0.13 % [−0.09 %] | +0.62 % [−0.44 %] | +0.10 % [−0.03 %] |
| GRU | +0.44 % [+0.27 %] | −0.44 % [−0.07 %] | −0.01 % [−0.17 %] |
| LSTM | +0.13 % [+0.34 %] | +0.96 % [+0.09 %] | +0.27 % [+0.06 %] |
| **Combined (served)** | **−0.34 % [−0.02 %]** | **−0.04 % [−0.21 %]** | **−0.29 % [−0.06 %]** |
| Stacked (experiment) | −0.72 % | +2.02 % | +0.06 % |

(Test percentages computed from `results/final_test_results.csv`; bracketed values from `results/walkforward_pooled.csv`.) No entry is significant on test. On the pooled record the only significant results are BTC GRU/LSTM being **worse**.

- **Best model for Bitcoin / Gold / Silver**: by validation (the honest selection) CatBoost / CatBoost / GRU. On test, none differs from the random walk. **There is no meaningful "best".**
- **One global model?** Not appropriate: the assets have different calendars, close times and features. Per-asset models with one shared *method* is correct.
- **Is the ensemble useful?** It is never the worst model, it is consistently inside the top half, and it removes the need to pick a winner by noise. It does not add skill.

---

## PHASE 13 — Ensemble audit

| Aspect | Combined (served) | Stacked (experiment) |
|---|---|---|
| Members | Ridge, RandomForest, LightGBM, CatBoost, GRU, LSTM | Ridge, LightGBM, CatBoost, GRU |
| Weighting | Equal (1/6), no fitting | Non-negative Ridge on standardised OOF predictions |
| Weights | — | BTC: LGBM 0.39, CatBoost 0.31, others 0 · Gold: CatBoost 1.18, GRU 0.59, Ridge 0.08 · Silver: GRU 0.99, LGBM 0.55 |
| Fitted on | nothing | walk-forward OOF of train+val (test never) |
| Test influence | none | none |
| Weak members hurt? | BTC: dropping GRU/LSTM would gain 0.09 % (E5a). Gold/Silver: no gain. | Weights > 1 on Gold over-amplify noise → worst Gold test model (+2 %) |

**Best individual vs ensemble:** on test the hindsight-best single model beats Combined by 0.4–1.1 %, but that model differs per asset and was never the validation pick. On the pooled record Combined is within 0.25 % of the best single model for every asset. **Keep Combined**: equal weighting is the robust choice when members are of indistinguishable quality (the "forecast-combination puzzle"), and it is honestly evaluated. The stack stays an experiment and is not served.

---

## PHASE 14 — Prediction behaviour (test window, Combined)

| | Bitcoin | Gold | Silver |
|---|--:|--:|--:|
| Actual return mean / std | +0.065 % / 2.08 % | −0.078 % / 1.68 % | −0.090 % / 3.20 % |
| Predicted return mean / std | −0.017 % / 0.12 % | +0.131 % / 0.09 % | +0.114 % / 0.09 % |
| Mean bias (pred − actual) | −0.08 % (under-predicts) | **+0.21 % (over-predicts)** | **+0.20 % (over-predicts)** |
| corr(predicted, actual) | +0.09 | +0.16 | +0.18 |
| Pooled walk-forward corr | +0.02 | +0.05 | +0.02 |
| UP calls / up days | 37 % / 49 % | 97 % / 50 % | 91 % / 52 % |
| Large moves (\|r\| > 2σ): mean \|actual\| vs \|predicted\| | 5.35 % vs 0.12 % | 3.92 % vs 0.21 % | 7.45 % vs 0.12 % |
| Large-move direction hit | 43 % | 25 % | 0 % |
| High-vol tercile RMSE: Combined vs RW | 2.315 vs 2.331 % | 1.942 vs 1.952 % | 3.609 vs 3.642 % |
| All 6 members agree on direction | 21 % of days | 83 % | 56 % |

Reading the table:
- **Do the predictions copy the latest price? Effectively yes.** The predicted move averages 0.09–0.14 % against actual moves of 1.3–2.5 %. The price line hugs the actual line only because it starts from today's close.
- **Too smooth:** prediction std is ~1/20 of actual.
- **Systematic over-prediction for the metals:** they learned the 2018–2025 bull-market drift and carried it into a 2026 window in which both metals fell.
- **Large movements are missed entirely**, by construction (the forecasts never exceed ±0.5 %).
- **Volatility:** in the high-volatility tercile Combined is marginally (0.5–0.9 %) below RW, which is not significant. The volatility-scaled *band* is what adapts to turbulence.
- The test-window correlations of +0.16/+0.18 look encouraging but are noise. On 1,584 pooled days they are +0.02 to +0.05.

---

## PHASE 15 — Each asset separately

**Bitcoin.** 3,147 feature rows (daily, 7 days/week), 29 features incl. Fear & Greed, day-of-week, volume and same-day S&P/VIX. Validation pick CatBoost (depth 7). Test: Combined −0.3 % vs RW (p 0.49), DA 46.9 %. The pooled record is the cleanest null (−0.02 %, p 0.87), and GRU/LSTM are significantly worse than RW. No drift problem (UP 37 %). Weakness: the most volatile asset, with the largest absolute $ errors (MAE ≈ $1,064). The only (non-adopted) improvement with a significant pooled gain is dropping the RNNs (−0.09 %), which is economically irrelevant.

**Gold.** 2,156 exchange days, 28 features with previous-day macro (DXY, 10-y yield, oil, S&P, VIX), lagged H/L features. Validation pick CatBoost (depth 3, 606 iterations). Test: Combined equal to RW (p 0.97), DA 50.3 %, UP calls 97 %. Pooled: −0.21 % (p 0.09), the closest to significance of all, but no better than the drift forecast (DM p vs drift 0.57). Weakness: drift dependence. Its forecast is essentially "gold's average daily rise".

**Silver.** 2,156 days, 29 features incl. Gold's same-day return. Validation pick GRU (9 epochs). Test: Combined −0.3 % (p 0.46), DA 53.8 %. LightGBM's 58.7 % (p 0.02) comes with DM p 0.98 and 86 % UP calls, and 1 of ~10 models at p < 0.05 is expected by chance. Weakness: the late-2025 volatility explosion (validation RMSE ~3× training) makes every volatility-based band under-cover on validation.

Same method, three assets, three different "best" models — and all three at the random-walk floor. That consistency is itself evidence that the null result is real.

---

## PHASE 16 — Prediction-reliability verdict

### 🟡 YES — ACCEPTABLE FOR FYP WITH LIMITATIONS

- **Not 🟢:** the point forecasts have no demonstrated skill. Baseline comparison on test and on 6 years of walk-forward data shows ±0.4 % of RW, with no significant improvement.
- **Not 🟠/🔴:** the predictions are *valid*. There is no leakage, no overfitting reaches the test, validation is time-aware and reproducible, metrics are correct, behaviour is fully explained (drift, near-zero amplitude), and the uncertainty band is calibrated (test coverage 66–69 % for 68 %). An honest, correctly evaluated null result on an efficient-market question is defensible. Inflating it would not be.

---

## PHASE 17 — Application audit

| Feature | Status | Notes (verified 2026-10-04) |
|---|---|---|
| Data download / collection | WORKING | Yahoo + alternative.me; complete-bar rule |
| Live sync | WORKING | Writes `data/raw_live`, atomic; network failure returns False (tested). Yahoo publishes BTC's previous-day bar late → a dashboard warning explains it. |
| Preprocessing | WORKING | Byte-identical re-run |
| Prediction (live and demo) | WORKING | All three assets |
| Model loading | WORKING | 18 models + scalers; a stale `model_status.json` raises a clear error |
| Ensemble (Combined) | WORKING | Exactly the mean of the members (tested) |
| Asset selection | WORKING | Sidebar + `?asset=` deep link |
| Charts | WORKING | Price + split shading, forecast + band, test actual-vs-predicted, return scatter, importance, comparison bars |
| Metrics | WORKING | Test, walk-forward, pooled, band, regimes |
| Streamlit UI | WORKING | 4 tabs, no exceptions, server log clean |
| Error handling | WORKING | NaN in newest row → error; missing artefacts → message; API 404/503/500 |
| Configuration | WORKING | One `config.py` |
| Saved models | WORKING | Git-ignored. A fresh clone needs `make collect-data preprocess train stack evaluate` (~15 min). |
| API | WORKING | `/health`, `/models`, `/predict/{asset}` incl. `?model=`; 404 for unknown asset or model |
| Documentation | NEEDS IMPROVEMENT | README/CHANGELOG/dashboard updated. **The PDF report's band paragraph is stale** (§23). |
| Installation | WORKING | `requirements.txt` pinned (Python 3.9 locally; CI runs 3.11 — not re-run in this audit) |
| Reproducibility | WORKING | Seeds; ≤ 1e-13 re-run |

---

## PHASE 18 — Bug audit

| Issue | Severity | Status |
|---|---|---|
| 68 % band covered 74–75 % on test (fat tails; mislabelled) | Medium (an honesty claim) | **Fixed**: calibrated k (§11) |
| Inference relied on the feature file's column order matching the scaler's | Medium (silent wrong forecast if violated) | **Fixed**: reorder by `scaler.feature_names_in_` + name check; new test |
| A stale `model_status.json` without band calibration would silently serve the wrong band | Low | **Fixed**: explicit `FileNotFoundError` + test |
| LightGBM `subsample=0.8` inert without `subsample_freq` | Low (misleading parameter, no effect on results) | **Documented** in code; not changed (it would alter served models without evidence of benefit) |
| README test count / band text, `src/README.md`, metrics docstring | Low | **Fixed** |
| PDF report / study guide describe the old band | Low | **Flagged** (§23); generators not in repo |
| Broken imports, wrong paths, stale model files, feature ordering, scaling, target handling, date errors, UI/backend mismatch | — | None found. Re-run reproduces all results; all tabs and the API work. |
| Cosmetic warnings (sklearn "X does not have valid feature names", pandas FutureWarning in one test) | Cosmetic | Left as is |

Retested after fixes: 45/45 tests pass; full evaluation re-run; dashboard (Gold, Bitcoin, Silver) and API checked.

---

## PHASE 19 — Final FYP quality

| Dimension | Assessment |
|---|---|
| Technical quality | High: modular pipeline, single data loader, explicit failure modes, tests, CI |
| ML methodology | High: leakage-audited, close-time rule, walk-forward tuning, two-phase training, multiple baselines, significance tests, pooled evidence |
| Prediction credibility | High *as a measurement*, null *as a forecast* |
| UI quality | Good: clear single forecast, honest reliability context, demo on unseen days |
| Code quality | Good |
| Documentation | Good. One PDF section to correct. |
| Reproducibility | Excellent |
| Presentation readiness | Ready (§21) |
| Viva readiness | Ready if the team owns the null-result narrative (§20, §22) |

---

## PHASE 20 — What we must NOT claim, and how to present it

**Never claim:** that the system predicts prices accurately; "1.3 % error = 98.7 % accuracy" (MAPE is the same for "tomorrow = today"); any profit or trading performance (the long/flat strategy returns are noise and below buy-and-hold for BTC); "deep learning outperforms" (GRU/LSTM are among the worst); that 52–59 % direction is skill (it is drift or chance); that the test set was never looked at; R² ≈ 0.95 on prices; market certainty.

**We can claim:** a leakage-audited, reproducible pipeline; a fair comparison of 9 models + 2 ensembles against 3 baselines on identical days; time-series-correct validation (walk-forward + held-out test + 6-year pooled record); the finding that next-day returns of BTC, Gold and Silver are statistically indistinguishable from a random walk with this data; that magnitude (volatility) *is* predictable, and that the served band is calibrated to ~68 % on unseen data; that we found and removed a real data leak (the 13:30 settlement) that had produced a fake 62 % result.

**Suggested framing:** *"We built a system to test whether machine learning can forecast next-day commodity prices from public daily data. We engineered it so that the test is fair — no future information, time-ordered validation, proper baselines and significance tests — and the answer is no: every model, including an ensemble, is statistically tied with 'tomorrow equals today'. That is consistent with efficient-market theory. What the data does predict is how big tomorrow's move is likely to be, and our dashboard shows that as a calibrated 68 % range."*

---

## PHASE 21 — Demonstration flow (verified in the running app, 2026-10-04)

1. **Open:** `make serve` → http://localhost:8501 (or `venv/bin/streamlit run app/streamlit_app.py`). The overview strip states the target, the served forecast and its test reliability.
2. **Select asset:** sidebar → Gold.
3. **Run prediction:** sidebar button. It refreshes the last complete bar (throttled to once per 30 min) and runs 6 models.
4. **Models execute:** the "What each model predicted" table (e.g. Gold, 2026-10-02 close $4,162.30: 6/6 UP).
5. **Ensemble:** final card "combination of 6 models": $4,169.62 ▲ +0.18 % for 2026-10-05.
6. **Final price + band:** 68 % band $4,120.23 – $4,219.60 (±1.19 % = 0.91 × σ 1.31 %), with its 69 % test coverage.
7. **Charts/metrics:** 60-day chart with forecast and band; reliability table vs random walk; *Model Performance* (test table, comparison chart, walk-forward and **long-run** tables, predicted-vs-actual, importance, band calibration, regimes).
8. **Unseen-day demo:** *Predict a Day* → pick a test day → **▶ Run models live** (a progress box shows each model running, then a green "● LIVE — computed at … in … s, N later days hidden" badge) → predicted vs actual close, error, HIT/FAIL, in/out of band, each model.
9. **Explain:** *Methodology* tab (target, data, split, features, close-time rule, models, combination, band, metrics).

Bitcoin and Silver were also run. Bitcoin shows the expected vendor-lag warning when Yahoo has not yet published the previous UTC day's bar. **Demo tip:** press *Run prediction* once before presenting so the TensorFlow models are warm (the first run takes ~15–20 s), and have the *Predict a Day* tab ready as a fallback if Wi-Fi fails (it uses the frozen dataset).

---

## PHASE 22 — Viva questions (based on the actual implementation)

### A. Basic (20)

1. **What does the project do?** — *Short:* Forecasts the next day's closing price of Bitcoin, Gold and Silver and honestly measures whether that forecast has skill. — *Detailed:* It downloads daily data from Yahoo Finance (2018 →), builds 28–29 backward-looking features, predicts the next-day log return with six ML models plus three baselines, combines the six into one forecast, converts it to a price with a calibrated 68 % band, and evaluates everything on time-ordered unseen data with significance tests. It is served through a Streamlit dashboard and a FastAPI.
2. **Why these three assets?** — *Short:* Three different market types. — *Detailed:* BTC is a 24/7 crypto market with a 00:00 UTC daily close; Gold and Silver are exchange-traded futures with a 13:30 ET settlement and a 5-day calendar. The same method has to handle different calendars, close times and drivers (sentiment vs dollar/rates).
3. **What exactly is predicted?** — *Short:* y_t = ln(P_{t+1}/P_t), the next day's log return. — *Detailed:* The models predict the return; the price shown is P̂_{t+1} = P_t·exp(ŷ_t). The target is standardised with the training mean and std for training, then converted back.
4. **What is the horizon?** — *Short:* One trading day. — *Detailed:* BTC: the next calendar day; metals: the next exchange day (`next_trading_day`). 5-day returns and volatility targets were explored as experiments (E3, E4).
5. **Where does the data come from?** — *Short:* Yahoo Finance via `yfinance`, plus alternative.me. — *Detailed:* BTC-USD, GC=F, SI=F OHLCV; DXY, WTI, VIX, 10-y yield, S&P 500 for macro; the crypto Fear & Greed index. All are free and need no key.
6. **What period?** — *Short:* 2018-01 to 2026-09-12. — *Detailed:* Train ≤ 2025-09-10, validation ≤ 2026-02-17, test to 2026-09-11/12. After warm-up rows are dropped, features start ~Feb 2018.
7. **Why predict returns, not prices?** — *Short:* Returns are stationary; prices are not. — *Detailed:* A model trained on price levels sees test prices outside its training range (gold went from $1,300 to $4,500). Returns have a stable distribution, and the random-walk baseline becomes explicit (ŷ = 0).
8. **What is the random-walk baseline?** — *Short:* "Tomorrow's price = today's price". — *Detailed:* Naive model, ŷ = 0. Under the efficient-market hypothesis it is the hardest baseline to beat at a daily horizon, so every model is tested against it with a Diebold–Mariano test.
9. **Which models are used?** — *Short:* Ridge, Random Forest, LightGBM, CatBoost, GRU, LSTM, plus Naive, Naive-Mean and ARIMA baselines and a stacked ensemble. — *Detailed:* They span linear, bagged trees, boosted trees and recurrent networks, all behind one interface in `src/models/registry.py`.
10. **What forecast does the user see?** — *Short:* "Combined": the equal-weight average of the six models' predicted returns. — *Detailed:* No weights are fitted, so nothing is chosen on test. The Combined forecast is evaluated exactly like every single model.
11. **Why not just serve the best model?** — *Short:* There is no reliably best model. — *Detailed:* Validation rankings differ by < 0.5 % and do not carry over to test (the best test model was never the validation pick). Picking a "winner" would be picking noise; averaging is the robust choice.
12. **What is the main finding?** — *Short:* No model beats the random walk. — *Detailed:* On the test window Combined is within ±0.3 % of RW RMSE (DM p 0.46–0.97), and on 1,584–2,328 pooled walk-forward days within 0.21 % (p ≥ 0.09). Directional hit-rates are at chance or at the drift level.
13. **Then is the system useless?** — *Short:* As a trading signal, yes; as a scientific measurement, no. — *Detailed:* It shows rigorously that next-day direction is unpredictable with public daily data, and it does provide a calibrated estimate of tomorrow's *range*, which is useful for risk.
14. **What does the dashboard show?** — *Short:* Forecast, demo on unseen days, model performance, methodology. — *Detailed:* Four tabs. Every forecast comes with test metrics vs the random walk, the UP-call share, the 68 % band and a disclaimer.
15. **What is "Predict a Day"?** — *Short:* A replay on the unseen test period. — *Detailed:* You pick a test day; the models use only data up to that day (frozen dataset), predict the next close, and the actual close is revealed with the error and HIT/FAIL. A unit test perturbs every future row to prove it cannot look ahead.
16. **What is the uncertainty band?** — *Short:* A 68 % range: P_t·exp(ŷ ± k·σ_t). — *Detailed:* σ_t is the EWMA volatility up to day t; k (0.83–0.91) is calibrated on training-period walk-forward errors. On the unseen test window it covered 66–69 % of days.
17. **Tech stack?** — *Short:* Python 3.9, pandas, scikit-learn, LightGBM, CatBoost, TensorFlow/Keras, statsmodels, Streamlit, Plotly, FastAPI, pytest. — *Detailed:* Versions are pinned in `requirements.txt`; GitHub Actions runs import checks and tests.
18. **How do you run it?** — *Short:* `make pipeline` then `make serve`. — *Detailed:* `make preprocess train stack evaluate` (~15 min with the committed tuned parameters), `make test`, `make serve` for Streamlit, `make api` for FastAPI, `make predict` to print forecasts.
19. **What does the API offer?** — *Short:* `/health`, `/models`, `/predict/{asset}`. — *Detailed:* `/predict` returns the Combined forecast with each member, the band (incl. `band_k`), test metrics and a disclaimer; `?model=GRU` selects one model; unknown asset or model → 404.
20. **Biggest lesson?** — *Short:* Evaluation design matters more than model choice. — *Detailed:* A same-day macro leak once produced a fake 62 % Silver accuracy. Fixing the close-time alignment removed it. Without strict time alignment and baselines, any model can look good.

### B. Machine learning (20)

1. **Is this classification or regression?** — *Short:* Regression of the next-day return; direction is derived from its sign. — *Detailed:* The loss is MSE on the standardised return. Directional accuracy is reported as a secondary metric with a binomial test.
2. **Why standardise the target?** — *Short:* Numerical stability and comparable scales. — *Detailed:* Daily returns are ~0.01; z-scoring with the *training* mean/std helps the neural nets and boosting. It is inverted exactly at prediction time.
3. **Why fit the scaler on train only?** — *Short:* To avoid leaking validation/test statistics. — *Detailed:* `MinMaxScaler().fit(train_df)`; val/test are only transformed. Fitting on all data would let test minima/maxima shape the inputs.
4. **Why do tree models use one row but RNNs use 30?** — *Short:* Trees need tabular inputs; RNNs model sequences. — *Detailed:* Tabular models get the features at day t (which already summarise the past via lags and rolling windows); GRU/LSTM get the last 30 days of the same features. Both predict the same target on the same days.
5. **What is two-phase training?** — *Short:* Choose the stopping point on validation, then refit on train+val. — *Detailed:* Phase A fits on train and monitors val for trees, iterations or epochs. Phase B refits on train+val with that number fixed, so the deployed model uses the most recent data and validation never monitors its own early stopping.
6. **Why do GRU/LSTM stop after 1–9 epochs?** — *Short:* There is no learnable pattern. — *Detailed:* Validation loss is lowest almost immediately; further training only fits noise. The resulting networks output nearly constant returns (prediction std ≤ 0.2 %).
7. **Is there overfitting?** — *Short:* Yes on train, no on test. — *Detailed:* LightGBM (BTC) reached train R² 0.335 and DA 71.5 % vs validation R² 0.04. Early stopping (81 trees) and walk-forward-selected regularisation stop it from reaching test (test R² +0.01).
8. **How does bias–variance apply?** — *Short:* With signal ≈ 0, any variance is pure error. — *Detailed:* The best possible forecast is close to a constant, so high-variance models are penalised. That is why the tuner chose strong regularisation (Ridge α = 100, shallow trees, high min-leaf).
9. **Why Ridge α = 100?** — *Short:* The walk-forward grid picked the strongest shrinkage. — *Detailed:* Grid {0.1, 1, 10, 100}. α = 100 won for all assets, which means shrinking coefficients towards zero, towards the random walk.
10. **What is walk-forward CV?** — *Short:* Train on the past, validate on the next block, expand, repeat. — *Detailed:* `TimeSeriesSplit(4)` on train+val: 4 consecutive validation blocks, each predicted by a model trained only on earlier data, with an inner 15 % tail for early stopping.
11. **Why not ordinary shuffled k-fold?** — *Short:* It lets the model train on the future. — *Detailed:* Shuffling puts future days in training and past days in validation; autocorrelated features then leak, which gives optimistic scores.
12. **What is the Diebold–Mariano test?** — *Short:* A test of whether two forecasts have equal accuracy. — *Detailed:* It uses the loss differential d_t = e_A² − e_B², tests mean(d) = 0 with a HAC variance and the Harvey–Leybourne–Newbold small-sample correction (`metrics.diebold_mariano`). p > 0.05 → no significant difference.
13. **How is directional accuracy tested?** — *Short:* A one-sided binomial test vs 50 %. — *Detailed:* Days with zero return are excluded. We also report the UP-call share, and in the pooled table we compare with the drift forecast's hit-rate, because "always UP" beats 50 % in a rising market.
14. **Why isn't price-level R² reported?** — *Short:* It is meaningless here. — *Detailed:* A random walk scores R² ≈ 0.95 on prices because today's price explains tomorrow's. Only return-space R² measures skill.
15. **What does return R² ≈ 0 mean?** — *Short:* The forecast explains none of the return variance. — *Detailed:* R² = 1 − SSE/SST; values in [−0.04, +0.03] on test mean the model is as good as predicting the mean return.
16. **What does feature importance tell you here?** — *Short:* What the model *used*, not what *works*. — *Detailed:* CatBoost relies on return_20d for Gold and ema14_ratio for Silver, but those relationships do not hold out-of-sample (pooled corr ≤ 0.05). Importance is not evidence of predictive value.
17. **Why does equal weighting work so well?** — *Short:* It averages out estimation noise. — *Detailed:* When members have similar accuracy, estimated weights add variance that outweighs their benefit (the "forecast-combination puzzle"). Our stack, with estimated weights, was the worst Gold model on test.
18. **What happened with stacking?** — *Short:* No gain. — *Detailed:* A non-negative Ridge on walk-forward OOF predictions gave weights such as CatBoost 1.18 + GRU 0.59 for Gold. That amplified noise: test RMSE +2 % vs RW. It is kept as a reported experiment.
19. **What is the "UP-call" problem?** — *Short:* The metals' forecasts are almost always UP. — *Detailed:* Trained on 2018–2025 bull markets, the models learn a positive mean, so the direction call is the drift. The forecast said UP on 97 % (Gold) and 91 % (Silver) of test days while ~50 % were up days, so the hit-rate ≈ the share of up days.
20. **Why does the band show skill when the forecast doesn't?** — *Short:* Volatility clusters; returns don't. — *Detailed:* Large moves follow large moves (ARCH effects), so EWMA volatility predicts the *size* of tomorrow's move. E4 shows volatility models beat persistence by 4–12 %, and the calibrated band covers 66–69 % on test.

### C. Dataset (15)

1. **How big is the dataset?** — *Short:* BTC 3,147 feature rows; Gold/Silver 2,156 each. — *Detailed:* Model samples: BTC 2,750 train / 160 val / 207 test; metals 1,874 / 109 / 143.
2. **Why start in 2018?** — *Short:* The Fear & Greed index starts in Feb 2018, and it gives ~8 years. — *Detailed:* E1 compared 3 years with the full history on the same validation window: mixed signs, no consistent gain.
3. **Why does BTC have more rows?** — *Short:* It trades 7 days a week. — *Detailed:* Missing BTC days are forward-filled (a data gap); futures keep exchange days only.
4. **Why no weekend rows for Gold/Silver?** — *Short:* They would be fake zero-return days. — *Detailed:* An early version forward-filled weekends, so ~30 % of rows were artificial zero returns. These were removed in the FYP-release fix.
5. **What is GC=F?** — *Short:* Yahoo's continuous COMEX front-month gold futures series. — *Detailed:* It follows the current front-month contract and can be revised after a roll (we saw a 1.4 % difference in the live download), so the frozen dataset is kept separate.
6. **What is the 13:30 settlement issue?** — *Short:* The metals' "close" is fixed before the US macro markets close. — *Detailed:* Using the same day's S&P/VIX/DXY returns would include information from after P_t. The metals therefore use the previous day's macro values, and their High/Low-based features are lagged a session.
7. **Which external series are used?** — *Short:* S&P 500 and VIX for all; DXY, oil and the 10-y yield for the metals; Fear & Greed for BTC; Gold's return for Silver. — *Detailed:* All enter as daily log changes (F&G as a level), aligned under the close-time rule.
8. **Is Fear & Greed safe to use same-day?** — *Short:* Yes. — *Detailed:* alternative.me publishes the day's value at 00:00 UTC, before that day's BTC bar closes at the next 00:00 UTC.
9. **What is the complete-bar rule?** — *Short:* Drop today's still-running bar. — *Detailed:* Yahoo returns the current day's partial bar as if it were final. We drop it: BTC until the UTC day ends, metals until 17:15 ET (`market_calendar.py`, tested).
10. **Why is futures volume not used?** — *Short:* It jumps at contract rolls. — *Detailed:* Yahoo's GC=F/SI=F volume is front-month only. For BTC we use the clipped log volume change.
11. **Frozen vs live data?** — *Short:* Frozen = evaluation; live = today's forecast. — *Detailed:* `data/raw` is never overwritten; live downloads go to `data/raw_live` and `*_live_features.csv`, so the reported results stay reproducible.
12. **What data-quality checks exist?** — *Short:* Dedup, positive prices, NaN/inf, minimum rows. — *Detailed:* `validate_features()` raises on any NaN/inf. A corrupted live download is rejected and never written.
13. **Are returns stationary?** — *Short:* Yes (ADF). — *Detailed:* `src/data/eda.py` runs ADF tests (prices non-stationary, returns stationary), ACF/PACF and distribution plots (fat tails, volatility clustering).
14. **Why are features "stationary"?** — *Short:* So that the test distribution matches training. — *Detailed:* Indicators are converted to ratios (P/EMA−1, MACD/P, ATR/P, %B) instead of price levels.
15. **Would more data help?** — *Short:* Not for direction. — *Detailed:* E1 showed no consistent gain from 2.9× more data. When the signal is ~0, more data mainly makes the "no skill" verdict more certain.

### D. Models (15)

1. **Ridge?** — *Short:* Linear regression with L2 penalty (α = 100). — *Detailed:* The simplest learner; a strong penalty shrinks it towards a constant forecast.
2. **Random Forest?** — *Short:* 300 trees, depth 3, min leaf 30 (all assets). — *Detailed:* Bagged trees; the walk-forward grid chose the shallowest, most regularised option.
3. **LightGBM?** — *Short:* Gradient boosting with leaf-wise trees, tuned leaves/learning rate/min-child/L2, early stopping. — *Detailed:* BTC used 81 trees, metals 10 (the floor). `subsample=0.8` is inert without `subsample_freq` (documented, no effect on results).
4. **CatBoost?** — *Short:* Boosting with symmetric trees; depth 7 (BTC) / 3 (Gold) / … — *Detailed:* Stopping points: 50 (BTC), 606 (Gold), 82 (Silver) iterations. The validation pick for BTC and Gold.
5. **GRU?** — *Short:* One GRU layer (32 units) → dropout → Dense(16) → Dense(1). — *Detailed:* Input: 30 days × 28–29 features; Adam lr 1e-3; batch 64; early stopping, patience 6. The validation pick for Silver.
6. **LSTM?** — *Short:* The same architecture with an LSTM cell, for a fair comparison. — *Detailed:* Among the worst models; significantly worse than RW for BTC on the pooled record.
7. **Why single-layer RNNs?** — *Short:* A bigger network overfits ~2,000 noisy samples. — *Detailed:* The original 2×100-unit stack (~130k parameters) overfitted badly; it was reduced and dropout was tuned (0.2/0.4).
8. **Why SEQ_LEN = 30?** — *Short:* About one trading month. — *Detailed:* Long-horizon information is already in the features (return_20d, rv_22d), and the RNNs stop within a few epochs regardless.
9. **ARIMA?** — *Short:* ARIMA(p,0,q), p,q ≤ 3 by AIC on the standardised returns. — *Detailed:* Forecasts walk-forward, one step ahead, appending each true value without refitting. It is a classical baseline and among the worst on test.
10. **Naive-Mean?** — *Short:* Always predict the training mean return. — *Detailed:* The "drift" forecast: always UP for an asset that rose. It is the right reference for directional accuracy.
11. **How were hyper-parameters chosen?** — *Short:* A grid search scored by walk-forward RMSE on train+val. — *Detailed:* Small grids focused on regularisation (`tune_models.py`), with the winners in `results/tuning/best_params.json`. The test set is never read.
12. **Why small grids?** — *Short:* Bigger searches overfit the validation folds. — *Detailed:* With differences of ~0.1 %, a large search would just find the luckiest configuration.
13. **How are models saved and loaded?** — *Short:* joblib `.pkl`, CatBoost `.cbm`, Keras `.keras`, plus JSON metadata. — *Detailed:* `registry.save/load`. Target statistics and the band k are in `model_status.json`, written by the evaluation.
14. **Which model is "best"?** — *Short:* By validation: CatBoost (BTC, Gold) and GRU (Silver); in reality none differs from RW. — *Detailed:* The ranking is not statistically decisive, which is why the served forecast is the combination.
15. **Why not XGBoost or Transformers?** — *Short:* No reason to expect a gain. — *Detailed:* XGBoost is in the same family as LightGBM/CatBoost; ExtraTrees and HistGB were tried (±0.2 %). Transformers need far more data and would only add variance when the RNNs stop after 1–9 epochs.

### E. Prediction (10)

1. **How is a prediction produced?** — *Short:* Features → scale → 6 models → average return → price. — *Detailed:* Unscaled features up to the last complete bar → train-fitted scaler (columns in training order) → tabular models on the last row, RNNs on the last 30 rows → de-standardise → mean of 6 → P_t·exp(ŷ) → band with k·σ_t.
2. **Why are predicted moves so small (~0.1 %)?** — *Short:* The models learned that tomorrow's move is unpredictable. — *Detailed:* The MSE-optimal forecast with no signal is the mean return; prediction std is ~1/20 of actual.
3. **Why does Gold almost always say UP?** — *Short:* Training-period drift. — *Detailed:* Gold rose strongly in 2018–2025; the models' intercept carries that. In the 2026 test window gold fell 10.6 %, so the forecast over-predicted by +0.21 %/day.
4. **How is the band computed?** — *Short:* ±k·σ_t around the predicted return. — *Detailed:* σ_t = EWMA (λ = 0.94) volatility at day t; k = the 68.27 % quantile of |error|/σ_t on training-period walk-forward errors (0.825 / 0.909 / 0.870).
5. **What if Yahoo hasn't published the latest bar?** — *Short:* The app forecasts from the last complete bar and warns. — *Detailed:* BTC's previous-day bar often appears hours after 00:00 UTC; the warning names the missing date and the as-of date used.
6. **Which date is the forecast for?** — *Short:* The next trading day after the last complete bar. — *Detailed:* BTC +1 calendar day; metals +1 business day (exchange holidays are not modelled; the label says "next trading day").
7. **Can it predict crashes?** — *Short:* No. — *Detailed:* On days with moves > 2σ the forecast averaged 0.1–0.2 % and got direction right 0–43 % of the time. Only the band widens when volatility is high.
8. **Do the models agree?** — *Short:* On direction only sometimes. — *Detailed:* All 6 agree on 21 % (BTC), 83 % (Gold) and 56 % (Silver) of test days. Gold's agreement is the shared drift.
9. **Why does the predicted line look accurate on the chart?** — *Short:* It starts from today's close. — *Detailed:* Any forecast near "no change" hugs the actual price line; skill is only visible in return space.
10. **Is this financial advice?** — *Short:* No. — *Detailed:* Every page and API response carries a disclaimer and shows the forecast's own test performance next to it.

### F. Validation (10)

1. **What are the split dates?** — *Short:* Train ≤ 2025-09-10, validation ≤ 2026-02-17, test after. — *Detailed:* Assigned by the date the target covers, identical for all assets (`config.py`).
2. **Why both a validation split and walk-forward CV?** — *Short:* CV for tuning; validation for stopping points and diagnostics. — *Detailed:* CV (4 folds on train+val) picks hyper-parameters; the fixed validation window picks trees/epochs in phase A.
3. **How is overlap prevented for multi-day targets?** — *Short:* An exact embargo. — *Detailed:* The split is by the target's last day, and CV purges h−1 samples before each validation block.
4. **Was the test set touched only once?** — *Short:* Read once per evaluation run by one script; inspected during development. — *Detailed:* No parameter or weight was ever fitted on it. But we looked at it during development (e.g. when the leak was found), so we also report the 6-year pooled walk-forward record, which agrees.
5. **What is the pooled walk-forward record?** — *Short:* All out-of-fold predictions combined. — *Detailed:* 2,328 (BTC) / 1,584 (metals) days, Oct 2019 → Feb 2026, about 10× the test window: Combined vs RW −0.02 / −0.21 / −0.06 %, DM p 0.87 / 0.09 / 0.69.
6. **Interpret DM p = 0.97 for Gold.** — *Short:* The Combined forecast and the random walk are indistinguishable. — *Detailed:* The squared-error differences average to essentially zero, and there is no evidence either is better.
7. **What about multiple testing?** — *Short:* With ~10 models × 3 assets, a few p < 0.05 appear by chance. — *Detailed:* Silver LightGBM's DA p = 0.02 is one of 30 tests; its DM p is 0.98. We do not claim isolated significant numbers.
8. **Does performance depend on the regime?** — *Short:* Slightly, not significantly. — *Detailed:* `regime_analysis.csv`: in the high-volatility tercile Combined is 0.5–0.9 % below RW RMSE; elsewhere equal. Regimes are defined on day t, so there is no look-ahead.
9. **How was leakage tested?** — *Short:* Unit tests plus a truncation test. — *Detailed:* Tests assert the target is never in the window, close-time alignment and complete bars; `predict_for_date` is re-run with every future row perturbed and the prediction is unchanged. In the 2026-09-21 audit, features rebuilt from truncated history were identical.
10. **Any genuinely new out-of-sample data?** — *Short:* Yes, a 15–20-day forward check. — *Detailed:* 2026-09-13 → 10-02 (after every design decision): RMSE within ±3.5 % of RW, DA 40–47 %. That is too few days for statistics, but consistent.

### G. Streamlit / code (10)

1. **How is the code organised?** — *Short:* `src/{data, models, training, evaluation, inference, api, utils, experiments}`, `app/`, `tests/`, `config.py`. — *Detailed:* Data flow: raw → preprocessing → `build_dataset` → tune/train/stack → backtesting → results → dashboard/API.
2. **Why one `build_dataset()`?** — *Short:* So that every stage sees identical samples. — *Detailed:* Tuning, training, stacking, evaluation and the experiments all call it, which prevents train/serve skew and split mismatches.
3. **Where is the test set read?** — *Short:* Only in `src/evaluation/backtesting.py`. — *Detailed:* Tuning and training never use `X_test`; `model_status.json` (the served metadata) is written there.
4. **How does the dashboard avoid slow reloads?** — *Short:* `st.cache_data` (5 min) and session state. — *Detailed:* The prediction result is cached per asset in `st.session_state`; the live download is throttled to once per 30 min.
5. **How are failures handled?** — *Short:* Explicitly, never silently. — *Detailed:* NaN or inf in features → ValueError; missing artefacts → FileNotFoundError with the command to run; non-finite model output → error; network failure → returns False and keeps the stored data; the API maps errors to 404/503/500.
6. **What is `model_status.json`?** — *Short:* The evaluation's record of the served setup. — *Detailed:* Members, test and pooled metrics, target mean/std, band k and coverage, split info, features. Inference reads the target statistics and k from it.
7. **What do the tests cover?** — *Short:* 45 tests. — *Detailed:* Feature look-ahead (crypto + futures), close-time alignment, complete-bar rule, target alignment, split integrity, price reconstruction, metrics, model contract, the Combined mean, no-look-ahead demo, the calibrated band, feature-order robustness, failure modes.
8. **How is reproducibility ensured?** — *Short:* Seeds, pinned versions, a frozen dataset. — *Detailed:* `set_all_seeds(42)` (Python/NumPy/TF), `random_state=42`, `shuffle=False`. The audit re-run reproduced every result file to ≤ 1e-13.
9. **How is live data kept from corrupting results?** — *Short:* Separate folders and an atomic write. — *Detailed:* `data/raw_live` and `*_live_features.csv` are written via a temp file + `os.replace`; the frozen files are never touched.
10. **What does CI do?** — *Short:* GitHub Actions: install, import checks, pytest. — *Detailed:* `.github/workflows/ci.yml` on push/PR to main, Python 3.11. Model-dependent tests skip when artefacts are absent.

### H. Difficult examiner questions (10)

1. **"Your model doesn't beat a naive guess. Why should this pass?"** — *Short:* Because the FYP's contribution is a correct measurement, and we demonstrate it. — *Detailed:* Next-day returns of liquid assets are close to a martingale (efficient-market theory). A project claiming 70 % accuracy would almost certainly be leaking, as our own early version did (62 % from a close-time leak). We built a leakage-audited, reproducible pipeline, compared 9 models and 2 ensembles against 3 baselines with significance tests, and produced a calibrated risk band where the data does have signal.
2. **"Silver LightGBM got 58.7 % direction with p = 0.02. Isn't that skill?"** — *Short:* No: it is one of ~30 tests, its DM p is 0.98, and it says UP 86 % of the time. — *Detailed:* Its RMSE equals the random walk's, so the hit-rate comes from drift and luck in a 143-day window. On the pooled record, Silver's best DA (53.5 %) is the "always UP" forecast.
3. **"Pooled Gold DA is 52.5 % with p = 0.023 — significant!"** — *Short:* Significant vs 50 %, but below "always UP" (55.8 %) on the same days. — *Detailed:* 55.6 % of those days were up. A forecast that is right less often than a constant UP call has no timing skill; the DM p vs drift is 0.57.
4. **"You've seen the test set many times. Isn't it contaminated?"** — *Short:* For design decisions, partly, and we say so. For parameters, no. — *Detailed:* No weight, stopping point or hyper-parameter was fitted on test. To guard against design-level snooping we report the 6-year pooled walk-forward record and a 15–20-day forward check. Both agree with the test, and the conclusion is negative, the opposite of what snooping produces.
5. **"Gold's same-day return is a feature for Silver, and Gold settles 5 minutes after Silver. Leakage?"** — *Short:* Technically 5 minutes; practically nothing. — *Detailed:* Those 5 minutes are ~0.36 % of a day's variance, which bounds the correlation of gold_return with Silver's next-day return at ≈ 0.003 (R² ≈ 1e-5). It is documented in `config.py` and `preprocessing.py`.
6. **"Why not use a Transformer or a bigger LSTM?"** — *Short:* The existing RNNs stop after 1–9 epochs: there is nothing to learn. — *Detailed:* Capacity is not the bottleneck; signal is. A bigger model on ~2,000 noisy samples only adds variance, and our LSTM is already significantly worse than RW for BTC.
7. **"You changed the band in the final audit. Isn't that tuning on the test?"** — *Short:* No: k was fitted on training-split walk-forward errors only, and the motivation came from the training folds. — *Detailed:* In 12/12 training folds the ±1σ band covered 73–76 % (fat tails). The test was evaluated after the decision and improved for all three assets (66–69 %). On the short validation window the result was mixed, and we report that too.
8. **"Hyper-parameters were tuned on the same folds you pool. Biased?"** — *Short:* Yes, in favour of the models, which makes the null result stronger. — *Detailed:* Even with that optimistic bias, no model beats the random walk on the pooled record.
9. **"MAPE is 1.3 % for Gold. Isn't that 98.7 % accurate?"** — *Short:* No: "tomorrow = today" also gets 1.29 %. — *Detailed:* MAPE on prices is dominated by the price level. The right comparison is the error relative to the random walk in return space, where the difference is 0.0 %.
10. **"If returns are unpredictable, what would you do with more time?"** — *Short:* Forecast volatility, or use richer data. — *Detailed:* E4 shows volatility is forecastable (Ridge beats persistence by 11.5 % for BTC). Making risk the served target, or adding intraday or order-flow data with strict timestamp alignment, are defensible next steps. More indicators on daily closes are not.

---

## PHASE 23 — Documentation status

Updated to match the final implementation: `README.md` (long-run table, calibrated band table, limitations incl. test-window reuse, 45 tests, structure), `CHANGELOG.md`, `results/FINAL_RESULTS.md` (auto-generated: new pooled and band sections), dashboard Methodology and Model Performance text, `src/README.md`, docstrings in `prediction.py`, `backtesting.py`, `metrics.py`, `audit_improvements.py`.

**Still to do (cannot be regenerated by code in the repo):**
- `docs/FYP_Technical_Report.pdf` (2026-09-14) describes the band as "±1 RMSE". That was already outdated before this audit (since 2026-09-21).
- `docs/FYP_Complete_Study_Guide.pdf` (2026-10-03) describes the band as ±1σ with 74–75 % test coverage.

Every *point-forecast* number in both PDFs is still correct (the forecasts did not change). Only the band paragraph needs replacing. Suggested erratum text:

> *Uncertainty band (final version).* Each forecast is shown with a 68 % band P_t·exp(r̂ ± k·σ_t), where σ_t is the RiskMetrics EWMA volatility (λ = 0.94) known at day t and k is the 68.27 % quantile of |r − r̂|/σ_t over the walk-forward errors of the training period (k = 0.83 Bitcoin, 0.91 Gold, 0.87 Silver; daily returns are fat-tailed, so k < 1). On the unseen test period the band contained the actual close on 65.7 % (Bitcoin), 69.2 % (Gold) and 69.2 % (Silver) of days, against 73.9 / 75.5 / 75.5 % for the uncalibrated ±1σ band and 79.2 / 42.0 / 44.8 % for a fixed-width band.

---

## PHASE 24 — Changelog of this audit (actual changes only)

| Change | Reason | Before | After | Improvement? |
|---|---|---|---|---|
| Pooled walk-forward significance (`backtesting.pooled_walkforward`, `results/walkforward_pooled.csv`, FINAL_RESULTS section, dashboard table, `model_status.combined_walkforward_pooled`) | One short test window has low power; need a drift reference for DA | Significance only on 143–207 test days | DM vs RW and vs drift + binomial DA on 1,584–2,328 days | Yes (evidence quality) |
| Band calibrated to 68 % (`backtesting.band_rows`, `prediction._band/_band_k`, `BAND_NOMINAL`) | ±1σ over-covered (fat tails) | ±1·σ_t, test coverage 73.9 / 75.5 / 75.5 % | ±k·σ_t, k from training errors; test 65.7 / 69.2 / 69.2 % | Yes on test; mixed on the short validation window |
| Fixed-width comparison band calibrated the same way | Fair comparison | Width = return std | Width = 68.27 % quantile of training errors | — (reporting) |
| `band_k`, `band_halfwidth_pct` in prediction dict + API schema | Transparency | — | Present | — |
| Missing band calibration → explicit error | No silent wrong band | Would serve ±1σ silently | `FileNotFoundError` | Yes (robustness) |
| Feature columns re-ordered to the scaler's order + name check | Silent misprediction risk | Count check only | Order- and name-safe | Yes (robustness) |
| LightGBM `subsample` inert — code comment | Honesty about parameters | Undocumented | Documented | — |
| `src/experiments/audit_improvements.py` + `results/experiments/E5_*` | Record the tested-and-rejected candidates | — | E5a/b/c reproducible | — (all rejected) |
| Tests: band test updated; +2 tests (missing calibration, column order) | Cover the changes | 43 | 45 | Yes |
| Docs: README, CHANGELOG, dashboard texts, `src/README.md`, Makefile `experiments` | Match the implementation | Stale band text, 43 tests | Updated | Yes |
| Live data refreshed (`data/raw_live`, git-ignored) | Forward check | Ended 2026-09-18/20 | Ends 2026-10-02 | — |

Not changed: data, features, target, splits, hyper-parameters, models, Combined rule, point forecasts.

---

## PHASE 25 — Before / after results (real measurements)

| Asset | Old performance (start of audit) | Final performance | Improvement |
|---|---|---|---|
| Bitcoin | Test RMSE (ret) 0.02072 vs RW 0.02079, DM p 0.49, DA 46.9 %; band coverage 73.9 % | **Identical point forecast**; pooled walk-forward −0.02 % (p 0.87); band coverage **65.7 %** (target 68.3) | Forecast: **none**. Band calibration: better on test, worse on validation (60.6 vs 75.0 %). |
| Gold | 0.01671 vs 0.01671, DM p 0.97, DA 50.3 %; band 75.5 % | Identical; pooled −0.21 % (p 0.09); band **69.2 %** | Forecast: none. Band: better on test and validation. |
| Silver | 0.03180 vs 0.03189, DM p 0.46, DA 53.8 %; band 75.5 % | Identical; pooled −0.06 % (p 0.69); band **69.2 %** | Forecast: none. Band: better on test, worse on validation (54.1 vs 57.8 %). |

**The prediction performance did not improve, because no candidate improved it on validation.** What improved is the strength of the evidence and the calibration of the band.

---

## PHASE 26 — Final verdict

### FYP FINAL STATUS: 🟡 READY AFTER FINAL MINOR FIXES

The minor fixes are non-technical: (1) **commit** the audit changes (`git add -A && git commit`), (2) replace the band paragraph in the PDF report with the erratum in §23 (or ask for the report to be regenerated), (3) rehearse the null-result narrative (§20). The code, models, evaluation and dashboard are final.

1. **Are our predictions good enough?** Good enough *for an FYP*, with limitations: they are valid and honestly evaluated, but have no forecasting skill.
2. **Are they better than the baseline?** No. Within ±0.3 % of the random walk on test (DM p 0.46–0.97) and within 0.21 % on 6 years of walk-forward data (p ≥ 0.09).
3. **Is there any leakage?** No leakage was found. The one historical leak (13:30 settlement) was fixed on 2026-09-13; a 5-minute Gold/Silver settlement gap is documented and negligible.
4. **Are we overfitting?** In-sample, yes (LightGBM BTC train R² 0.34). Out-of-sample, no: early stopping and regularisation keep every test result at the random-walk level.
5. **Is our validation trustworthy?** Yes: chronological, walk-forward, train-only scaling, reproducible, and confirmed by a 10× larger pooled record. Caveat: the test window was inspected during development.
6. **Is our ensemble actually useful?** As a robust, selection-free default, yes; as a source of extra accuracy, no. The fitted stack is worse, and Combined is never far from the best single model.
7. **Which asset performs best?** None meaningfully. Gold comes closest on the pooled record (−0.21 %, p 0.09), but no better than the drift forecast.
8. **Which performs worst?** Bitcoin on the pooled record (−0.02 %, p 0.87, with its RNNs significantly worse than the random walk). Gold on the test window (exactly equal to RW, DM p 0.97).
9. **Biggest remaining weakness?** No point-forecast skill, plus metals' forecasts that are mostly training-period drift (UP on 91–97 % of test days).
10. **Biggest strength?** Methodological integrity: leakage-audited, time-correct, reproducible to 1e-13, with honest significance testing that caught and removed a fake 62 % result.
11. **Final 3 most important improvements (for the future, not required now):** (a) make volatility/risk the served forecast (it *is* predictable, as E4 shows); (b) a longer, never-inspected forward test (collect daily predictions for 6+ months); (c) richer, correctly timestamped data (intraday/order-flow) instead of more daily indicators.
12. **Anything technically risky for the viva?** Three things to prepare: test-set reuse (answer: pooled + forward evidence), the band change made in the final audit (answer: k fitted on training errors only), and isolated "significant" numbers such as Silver LightGBM's 58.7 % (answer: multiple testing, DM p 0.98, drift). The outdated band paragraph in the PDFs is the only documentation risk.
13. **Can we confidently present this project to the university?** **Yes**, provided it is presented as what it is: a rigorous, honest test of next-day predictability with a working forecasting application, not a profitable predictor.
