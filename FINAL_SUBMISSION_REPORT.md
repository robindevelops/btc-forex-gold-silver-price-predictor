# Final Submission Report — AI-Powered Commodity Price Predictor (Bitcoin · Gold · Silver)

*Submission audit, 2026-10-10. Every number below was read from the project's result files or recomputed by its code today; nothing is estimated. The previous audit (2026-10-04, 110 viva Q&A) is `FINAL_AUDIT.md`.*

## 1. Final verdict

**Ready with clearly identified minor limitations.**

- The pipeline is leak-free as far as could be tested (target = next row, close-time rule for the metals, train-only scaler, chronological split, truncation and perturbation tests) and **bit-reproducible**: re-running the full evaluation today reproduced every stored number to ≤ 9·10⁻¹⁶, and a clean copy of the versioned files rebuilds every processed file byte-identically.
- The dashboard, API and CLI work for all three assets; 55 automated tests pass.
- Two real live-data bugs and a reproducibility gap were found and fixed today (section 2).
- The scientific result is unchanged and honest: **no model forecasts next-day returns of Bitcoin, Gold or Silver significantly better than the random walk** — on the 143–207-day test window, on 1,584–2,328 pooled walk-forward days, and on the 20–27 days after the dataset was frozen. The strongest genuine contribution is the rigorous, reproducible evaluation that establishes this, plus a calibrated uncertainty band (volatility *is* predictable).
- Minor limitations remain (section 7); none blocks submission. Two need your action: commit the work, and decide whether to amend the PDFs (they are not wrong, only missing today's additions).

## 2. What was wrong

| # | Issue | Type · severity | Cause | Fix (verified) |
|---|---|---|---|---|
| 1 | A fresh clone could neither run nor reproduce anything | Reproducibility bug · **High** | `data/raw/` (frozen dataset) and `data/models/` (trained models) were git-ignored; the README's recovery path re-downloads from Yahoo, which revises history and rolls futures contracts | Both folders are now versioned (≈ 3 MB). In a clean copy of the versioned files: `make preprocess` rebuilt all 12 processed files + 3 scalers **byte-identically**, `make predict` worked, 54 tests passed (1 skipped: needs live data) |
| 2 | One failed external download could break every forecast | Bug · Medium | The live sync wrote a feature file without the failed series (e.g. Fear & Greed); inference then failed with "Feature mismatch" until a later sync | `update_live_data` refuses to write a file lacking a model feature (`missing_model_features`); test added |
| 3 | Silver's live forecasts used a stale gold return | Bug (silent wrong input) · Medium | Silver's gold download failure was ignored, so the newest Silver day used yesterday's `gold_return` (observed today: 2026-10-09 got 0.39 % instead of 1.42 %) | Live sync is now **all-or-nothing** (every series an asset needs, gold for Silver, must be downloaded in that call; gold's last bar must not be older than Silver's); affected file rebuilt; test covers all failure modes. The frozen evaluation data were never affected |
| 4 | API/CLI silently served stale forecasts (Silver from 2026-10-02 data) | Transparency gap · Low | Only the dashboard refreshes data | `data_is_current` in every forecast and the API; `make predict` warns |
| 5 | Predict-a-Day said "(frozen dataset)" for NEW live days, and their markers floated past the end of the chart | Display bug · Low | Hard-coded caption; chart drew only the stored period | Caption shows the real data source and "after the stored test period"; newer live closes drawn (verified in the browser) |
| 6 | Text errors: Methodology "data from 2018-03-01" (that is the first *training sample*); Fear & Greed/day-of-week/volume listed as "macro returns"; "Stand on any day after 2026-02-17" while 02-17 was selectable; sync warning "Using the stored dataset" (wrong); notebook 01 "three years of data" (it is 8.7) | Text · Low | Stale/imprecise wording | Corrected; notebooks re-executed with 0 errors |
| 7 | "Untouched test set — evaluated once" | Methodological wording · Low | The test window was inspected across project iterations (leak fix, Combined forecast, band) — the README already admitted this | Headings now say what is true: held out from training, tuning and model selection; the pooled walk-forward and post-freeze evidence carry the larger-sample argument |
| 8 | Walk-forward folds use the scaler/target statistics fitted on the whole training split | Methodological weakness · Low | Single train-fitted scaler reused inside CV | Documented (README). Affects only model selection, pooled evidence and band k, only for Ridge/GRU/LSTM (trees are invariant); any bias favours the models; the test evaluation is unaffected |
| 9 | No moving-average baseline (you asked for one) | Missing comparison · Optional | — | `MA-5` and `Mean-20d` added to every table (section 3) |

Checked and found **correct** (no change needed): target construction and alignment, chronological split by target dates, train-only scaler, close-time rule (metals use previous-day macro and lagged High/Low), feature look-ahead (perturbation/truncation tests), live == evaluation path, Combined = mean of members, band calibration on training-split errors only, complete-bar rule, data quality (no nulls, duplicates, unsorted or non-positive prices; BTC has no missing days; extreme days are genuine events — BTC −37 % on 2020-03-12, WTI −$37.63 on 2020-04-20, silver −31 % / gold −11 % on 2026-01-30 — and were kept). The 5-minute overlap of gold's 13:30 settlement with silver's 13:25 one is a documented approximation; dropping the whole macro group (including `gold_return`) changes Silver's walk-forward RMSE by ≤ 0.05 % (E2), so it cannot be inflating anything.

## 3. Accuracy investigation

**Target:** next-day log return `y = ln(P_{t+1}/P_t)`, price forecast `P̂ = P_t·exp(ŷ)`. Regression, so "accuracy" is reported as MAE/RMSE/MAPE plus directional hit-rate with a binomial test; MAPE alone is not accuracy (the random walk has the same MAPE). **Splits:** train ≤ 2025-09-10, validation ≤ 2026-02-17, test = 2026-02-18 → 2026-09-12 (Bitcoin, 207 days) / → 2026-09-11 (Gold, Silver, 143 days). Models: Ridge, Random Forest, LightGBM, CatBoost, GRU, LSTM; served forecast = their equal-weight mean ("Combined").

Test-set RMSE of the next-day return relative to the random walk (negative = better), same days for every row (`results/final_test_results.csv`):

| | Bitcoin | Gold | Silver |
|---|--:|--:|--:|
| Random walk (persistence) RMSE (return) | 0.02079 | 0.01671 | 0.03189 |
| Random walk MAE (USD) · MAPE | $1,066.42 · 1.52 % | $58.26 · 1.29 % | $1.79 · 2.49 % |
| Drift (training-mean return) | −0.05 % | +0.26 % | +0.10 % |
| ARIMA | +0.56 % | +0.62 % | +1.14 % |
| **MA-5** (5-day average price) — new | **+57.0 %** | **+52.3 %** | **+55.5 %** |
| **Mean-20d** (20-day mean return) — new | +2.68 % | +4.17 % | +5.73 % |
| Best single model by walk-forward CV | CatBoost +0.13 % | CatBoost +0.62 % | GRU −0.01 % |
| **Combined (served; original = final)** | **−0.34 %** (DM p 0.49) | **−0.04 %** (p 0.97) | **−0.29 %** (p 0.46) |
| Combined MAE (USD) · MAPE | $1,063.94 · 1.52 % | $58.20 · 1.29 % | $1.78 · 2.47 % |
| Combined direction correct (p) · UP calls / up days | 46.9 % (0.83) · 37 % / 49 % | 50.3 % (0.50) · 97 % / 50 % | 53.8 % (0.20) · 91 % / 52 % |
| Pooled walk-forward, Combined vs RW (days, DM p) | −0.018 % (2,328, 0.87) | −0.213 % (1,584, 0.09) | −0.058 % (1,584, 0.69) |
| Post-freeze 2026-09-13 → 10-09, Combined vs RW (days, DM p) | −0.05 % (27, 0.96) | +2.43 % (20, 0.32) | +0.67 % (20, 0.45) |

**Improvement attempts today (E6, walk-forward folds only, decision rule fixed before running — adopt only if better for every asset with DM p < 0.05 vs the served forecast):** volatility-scaled target (weighted least squares) for the tabular members: +0.165 % *worse* for Bitcoin (p = 0.016), +0.031 % Gold (p = 0.70), −0.053 % Silver (p = 0.42). Huber loss for LightGBM/CatBoost: +0.010 % / +0.034 % / +0.015 % (all p > 0.5). **Neither adopted; the served models are unchanged.** Together with E1–E5 (data size, feature ablation, horizon, volatility target, ensemble membership, de-duplicated features, zero drift — all rejected) this is a thorough, pre-registered search, and it consistently finds the random-walk floor.

**Per-asset decision.** Bitcoin, Gold, Silver: keep the equal-weight Combined of the six models (the best single model by CV differs per asset — CatBoost, CatBoost, GRU — and no ranking is significant, which is exactly when equal weights are the robust choice). No asset shows a demonstrated advantage over naive forecasting. The 68 % band (k·σ_t, k = 0.825 / 0.909 / 0.870) is the one calibrated, useful output: test coverage 65.7 / 69.2 / 69.2 % against a nominal 68.3 %.

**Why accuracy cannot be improved here.** Daily returns of liquid assets are close to a martingale: their autocorrelation is ≈ 0 (EDA), any persistent predictable component is arbitraged away, and the noise (σ ≈ 1.7–3.2 % per day) dwarfs any signal the models find (actual daily moves are 17–37 times larger than the models' predicted moves: σ 2.08 / 1.67 / 3.19 % vs 0.12 / 0.09 / 0.09 %). The pooled walk-forward test would detect an out-of-sample R² of about 0.5 % (t ≈ √(n·R²) ≈ 2.8–3.4 for 1,584–2,328 days); the observed values are ≈ 0, so any real signal is smaller than that. The tuned models learn this — they shrink towards "no change" — so their errors equal the random walk's. Chasing a better number on the test window would be test-set snooping (e.g. a zero-drift forecast would look better on 2026's falling metals; E5 rejected it on validation).

## 4. What was improved

| Change | Why | Verification |
|---|---|---|
| `data/raw/`, `data/models/` versioned; README fresh-clone quick start (`make preprocess` → `make serve`) | Reproducibility from GitHub | Clean-copy simulation: byte-identical rebuild, predict OK, 54 tests pass |
| All-or-nothing live sync + feature check (`src/data/sync_live_data.py`) | Bugs 2 and 3 | New test (4 failure/success paths); Silver live file rebuilt and checked (gold_return 1.42 %) |
| `data_is_current` (`prediction.py`, API, CLI) | Stale forecasts were silent | API/CLI output checked |
| MA-5 and Mean-20d baselines (`registry.rolling_baseline_forecast`, CV, backtesting, pooled table) | Requested; shows why persistence is the benchmark | 7 tests (definition, no look-ahead, stored rows = definition); full evaluation re-run — all older numbers identical to ≤ 9·10⁻¹⁶ |
| Cross-asset baseline table, honest chart captions, multiple-testing note, E5/E6 tables (dashboard) | Evaluators see the comparison at a glance; the zoomed chart no longer silently omits anything | Clicked through all tabs for all assets; no server or browser errors |
| Predict-a-Day caption/chart, Methodology text, sync warning, notebook text | Display/text bugs 5–6 | Browser check; notebooks re-executed, 0 errors |
| E6 experiment (`src/experiments/final_candidates.py`, `results/experiments/E6_*`) | Genuine, principled accuracy attempts | Ran to completion; decision recorded |
| README, CHANGELOG, `FINAL_RESULTS.md` header, FINAL_AUDIT pointer, this report | Documentation matches the verified state | README tables generated from the CSVs (an earlier hand-typed draft was caught and corrected) |

## 5. Why this project stands out

1. **It answers its research question rigorously and honestly.** Thirteen forecasting methods (three classical baselines, two moving-average rules, four ML models, two recurrent nets, a stack and an equal-weight combination) on identical days, with Diebold–Mariano tests, binomial direction tests, drift-aware direction analysis (UP-call share) and three independent evidence windows (test, pooled walk-forward, post-freeze). The conclusion — no skill over the random walk at a one-day horizon — agrees with the efficient-markets literature and is supported, not asserted.
2. **It found and fixed its own leaks.** A 62 % Silver "result" was traced to a close-time leak (same-day macro data against a 13:30 ET settlement), fixed, and archived — a subtle timing error that is easy to miss. Leak-safety is tested (perturbing future rows must not change any feature or prediction).
3. **It shows that baseline choice decides the headline.** Against a 5-day moving average the models would "win" by > 50 %; against persistence they tie. Reporting both is a methodological contribution in itself.
4. **It forecasts what *can* be forecast.** The calibrated, volatility-conditional 68 % band covers 65.7–69.2 % of unseen days (nominal 68.3 %), where a naive ±1σ band over-covers (74–76 %) and a fixed band fails (42–45 % for the metals).
5. **It is reproducible end to end** — from a fresh clone, bit-for-bit — and engineered like a product: complete-bar rule, all-or-nothing live data, freshness flags, an API, a dashboard that shows every forecast next to its own track record and the random walk, and 55 tests.

How to defend moderate accuracy: *"A model that cannot beat 'tomorrow = today' should say so. Our contribution is a pipeline that measures this correctly — no leakage, chronological validation, significance tests, the right baseline — and a dashboard that never presents a number without its evidence."*

## 6. Testing results

| Check | Command | Result |
|---|---|---|
| Full test suite | `venv/bin/python -m pytest tests/ -q` | **55 passed** (47 before today) |
| Clean-copy tests | `make test` in a copy of only the versioned files | 54 passed, 1 skipped (needs a live download) |
| Syntax/imports | `ast.parse` of every edited file; module imports | OK |
| Data audit | scratch script over `data/raw/*.csv` | no nulls/duplicates/unsorted/non-positive asset prices; BTC 0 missing days; extremes verified as genuine |
| Full evaluation re-run | `python src/evaluation/backtesting.py` | exit 0; every pre-existing row and `model_status.json` equal to ≤ 9·10⁻¹⁶ (last-digit float rounding) |
| Figures | `python src/evaluation/plots.py` | all PNGs byte-identical to the committed ones |
| Preprocessing reproducibility | `make preprocess` in the clean copy | 12 processed files + 3 scalers byte-identical |
| E6 experiment | `python src/experiments/final_candidates.py` | exit 0; no candidate adopted |
| Notebooks | `jupyter nbconvert --execute --inplace notebooks/*.ipynb` | 4/4 executed, 0 error outputs |
| Dashboard | Streamlit in the browser: Bitcoin, Gold, Silver — Run prediction, live scorecard, Predict a Day (stored and NEW days), Model Performance, Methodology, experiments expander | all work; no server/console errors |
| API | `/health`, `/models`, `/predict/{asset}`, `?model=GRU`, unknown asset (404), unknown model (404) | all correct |
| Live sync | `update_live_data` for all assets | works; during a Yahoo throttling window it returned False and kept the old files, as designed |

Not tested: Windows/Linux (macOS only; CI config targets Ubuntu + Python 3.11 but was not run — this Mac uses Python 3.9.6); the full re-tuning (`make tune`, ~30 min; tuned parameters reused); an offline start of the dashboard.

## 7. Remaining issues (most severe first)

1. **Uncommitted work** — everything above is in the working tree; commit it (section 8). Uncommitted work in this repo has been lost before.
2. **PDFs are not updated** — `docs/FYP_Technical_Report.pdf`, `docs/FYP_Complete_Study_Guide.pdf` and the Phase-2 thesis (Desktop) do not mention the MA-5/Mean-20d baselines, E6, or today's fixes, and the two `docs/` PDFs still describe the earlier ±1σ / ±1 RMSE band. All their *numbers* remain valid (reproduced today). If an erratum is wanted: "The 68 % band is P_t·exp(r̂ ± k·σ_t), k calibrated on training-split walk-forward errors (0.825 / 0.909 / 0.870); test coverage 65.7 / 69.2 / 69.2 %."
3. **The test window is not pristine** — inspected across iterations; disclosed. Larger-sample pooled and post-freeze evidence agree.
4. **Walk-forward folds share the training-split scaler** — documented; bias would favour the models.
5. **Live mode depends on Yahoo** — throttling makes syncs fail intermittently (handled safely); futures roll differences between live and frozen data.
6. **Multiple testing** — dozens of p-values; e.g. MA-5's 52.1 % Bitcoin direction hit-rate (p = 0.022, walk-forward) does not survive correction and is n.s. on test (52.7 %, p = 0.24). A pre-registered test of short-term mean reversion is future work, not a finding.
7. **Vendor artefacts** — two flat Gold bars (2026-01-27/28, H = L = C) inside validation; kept, documented.
8. **Untracked `docs/FYP_Phase2_Final_Study_Guide.pdf`** (created 2026-10-09, not by this audit) — decide whether to commit it.

## 8. Final run instructions

```bash
cd ~/Desktop/crypto-forex-prediction-system
python3 -m venv venv && source venv/bin/activate        # this Mac: the existing venv works (Python 3.9.6)
pip install -r requirements.txt
make preprocess          # rebuild data/processed from the versioned frozen data (~10 s)
make test                # 55 tests
make serve               # dashboard at http://localhost:8501
make api                 # optional: API at http://localhost:8000/docs
make predict             # optional: print today's forecasts
make evaluate            # optional: reproduce every result file (~15 min)
```

Do **not** run `make collect-data` to reproduce the report (it overwrites `data/raw/` with a new Yahoo download; `git checkout data/raw` restores it). To commit today's work:

```bash
git add -A && git commit -m "Submission audit: reproducible clone, MA baselines, live-data fixes, E6"
```

## 9. Viva defence

1. **Problem?** Forecast tomorrow's close of Bitcoin, gold and silver futures from public daily data — and measure honestly whether ML beats the naive "tomorrow = today" forecast.
2. **Why these assets?** Three different markets: a 24/7 crypto asset, a macro safe-haven and an industrial/precious hybrid; different calendars (7-day vs exchange days), volatilities (1.7–3.2 %/day) and drivers — a fair test of generality.
3. **What exactly is predicted?** The next trading day's log return; the price is P_t·exp(r̂). One-day horizon. Returns are stationary and make the random walk (r̂ = 0) explicit.
4. **Data and limits?** Yahoo Finance daily OHLCV 2018-01 → 2026-09-12 (BTC 3,177 rows, metals 2,186) + S&P 500, VIX, DXY, WTI, 10-y yield, Fear & Greed. Limits: one vendor; continuous futures roll; daily bars only; Yahoo revises history (hence the frozen, versioned copy).
5. **Why these models?** A ladder of complexity on the same inputs: Ridge (linear), Random Forest/LightGBM/CatBoost (non-linear trees, strong on tabular data), GRU/LSTM (sequence models over 30 days), ARIMA (classical), plus naive baselines. If the simplest model ties the most complex, that is evidence about the data, not the code.
6. **Features?** 28–29 stationary, backward-looking features: current and lagged returns, 20-day momentum, rolling/HAR/EWMA volatility, RSI, ADX, ROC, MACD/price, EMA ratios, Bollinger %B/width, ATR/price, high–low range, macro returns, sentiment/calendar for BTC, gold's return for silver. Price levels are never inputs (they leave the training range).
7. **Leakage?** Target is the next row and never in the window (tested); every feature uses only data known at the close (perturbing future rows changes nothing — tested); metals use previous-day macro data and previous-session High/Low because their close is the 13:30 ET settlement; scaler fitted on train only; split by target dates; test read by one script; the live code path reproduces the stored evaluation predictions (≤ 3·10⁻¹⁰, re-checked today).
8. **Why chronological validation?** Shuffled CV lets the model train on the future of the days it is tested on — leakage through overlapping rolling features and volatility regimes. Expanding-window walk-forward mimics real use: always predict forward from the past.
9. **Metrics?** RMSE/MAE of the return (what is predicted), MAE/RMSE/MAPE in USD (what users see), direction hit-rate with a binomial test and the share of UP calls (drift), Diebold–Mariano test vs the random walk, band coverage. R² on price levels is not reported — a random walk scores ≈ 0.95 there.
10. **Versus the naive baseline?** Ties it: −0.34 / −0.04 / −0.29 % test RMSE, DM p 0.49 / 0.97 / 0.46; pooled walk-forward DM p 0.87 / 0.09 / 0.69. The moving-average baselines are 50 %+ worse — beating them means nothing.
11. **Results per asset?** Section 3 table. MAPE 1.52 % (BTC), 1.29 % (Gold), 2.47 % (Silver) — within 0.02 percentage points of the random walk's.
12. **Does the ensemble help?** A little, never significantly. By convexity, the squared error of an average is never worse than the average of its members' squared errors, and equal weights avoid crowning a winner from noise: the CV-selected single model (CatBoost / CatBoost / GRU) did worse on the test window than the Combined (+0.13 / +0.62 / −0.01 % vs −0.34 / −0.04 / −0.29 %). On the test window the Combined ranks 3rd / 3rd / 2nd of the eight learned forecasts (Random Forest was best for Gold and Silver, the stack for Bitcoin) — none of these differences is significant, and picking the test-window winner would be snooping. The fitted stack is unreliable (best for Bitcoin, −0.72 %, but +2.0 % worse for Gold).
13. **Why is this hard?** Efficient markets: predictable returns get traded away; actual daily moves are 17–37× larger than the models' predicted moves; regimes shift (2026's metals crash); fat tails dominate squared errors.
14. **Contributions?** Section 5: leak-audited, reproducible evaluation; baseline lesson; calibrated volatility band; honest product.
15. **Limitations?** Section 7.
16. **What happens on "Run prediction"?** Check for a newer complete bar → download asset + required external series (all-or-nothing) → rebuild features → scale with the train-fitted scaler → last 30 rows → six models → average r̂ → price and 68 % band from σ_t → live scorecard re-runs every model on the last 20–90 days and compares with "tomorrow = today".
17. **Streamlit ↔ backend?** `app/streamlit_app.py` imports `src.inference.prediction` (forecasts), `src.data.sync_live_data` (downloads) and reads `results/*.csv` and `data/models/model_status.json` written by `src/evaluation/backtesting.py`; the API calls the same `predict_next_day`.
18. **Next steps?** Forecast volatility and risk (where skill exists) instead of direction; intraday or order-book data; pre-registered tests of short-term mean reversion; probabilistic forecasts scored with CRPS; a second data vendor.

Hard questions: *"Your model is useless then?"* — As a point forecaster, yes, and we prove it; the band and the evaluation framework are useful. *"Why not deep learning/Transformers?"* — Our GRU/LSTM were no better than the trees, and for Bitcoin they were the worst learned models (significantly worse than the random walk on the pooled walk-forward days, p = 0.04 / 0.004); more capacity on ~2,000 noisy samples over-fits rather than finds signal. *"Isn't 53.8 % direction for Silver good?"* — It said UP on 91 % of days and 52 % were up days; the hit-rate is drift, p = 0.20. *"Did you tune on the test set?"* — No: tuning on walk-forward folds inside train+val; the test window was inspected across iterations, which we disclose, and any snooping would favour the models — they still don't win.

## 10. Final submission checklist

| Item | State |
|---|---|
| Source code | ✅ Updated and tested — **you must commit** |
| Model artefacts | ✅ `data/models/` complete, versioned (after commit) |
| Dataset | ✅ `data/raw/` frozen dataset versioned (after commit); processed data rebuilt by `make preprocess` |
| Application | ✅ Dashboard and API verified for all assets |
| Results | ✅ `results/` regenerated; older numbers identical |
| Tests | ✅ 55 passing |
| README / CHANGELOG / FINAL_RESULTS | ✅ Updated |
| PDFs (docs/, thesis on Desktop) | ⚠️ Your decision: valid numbers, missing today's additions; band erratum in section 7 |
| `docs/FYP_Phase2_Final_Study_Guide.pdf` | ⚠️ Untracked — your decision |
| Backup | ✅ Pre-audit snapshot `~/Desktop/FYP_backup_2026-10-10/data_and_results_before_final_audit.tgz` |

## 11. Files changed

Project: `~/Desktop/crypto-forex-prediction-system`.

- **Created:** `FINAL_SUBMISSION_REPORT.md`, `src/experiments/final_candidates.py`, `tests/test_baselines.py`, `results/experiments/E6_final_candidates.csv`, `results/experiments/E6_summary.md`.
- **Edited:** `.gitignore`, `Makefile`, `README.md`, `CHANGELOG.md`, `FINAL_AUDIT.md` (pointer), `app/streamlit_app.py`, `src/models/registry.py`, `src/evaluation/cross_validation.py`, `src/evaluation/backtesting.py`, `src/evaluation/plots.py`, `src/inference/prediction.py`, `src/api/app.py`, `src/data/sync_live_data.py`, `tests/test_models_and_inference.py`, `notebooks/0[1-4]_*.ipynb` (re-executed; 01/04 text).
- **Regenerated:** `results/final_test_results.csv`, `cv_results.csv`, `walkforward_pooled.csv`, `FINAL_RESULTS.md`, `predictions/*_test_predictions.csv` (+ baseline rows/columns; all other values unchanged); `band_calibration.csv`, `regime_analysis.csv`, `model_status.json`, figures (byte-identical).
- **Newly versioned (were git-ignored):** `data/raw/*.csv`, `data/models/*`.
- **Live data refreshed** (git-ignored): `data/raw_live/`, `data/processed/*_live_features.csv`.
- **Removed:** nothing.
