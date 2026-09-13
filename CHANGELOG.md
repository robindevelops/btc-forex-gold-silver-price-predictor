# Changelog

## [Final audit fix] — 2026-09-13 — close-time leak for the metals removed, pipeline re-run

- **Leak found and fixed (critical).** Yahoo's daily close for GC=F / SI=F is the 13:25–13:30 ET COMEX settlement (verified against 5-minute contract bars), so the same-day S&P 500 / VIX / 10-y yield / DXY / WTI returns and the session High/Low used as day-*t* features contained information from inside the target interval. Commodity assets now use the latest external value **strictly before** day *t* (`config.EXTERNAL_SAME_DAY`, `DataCleaner._align_external`) and lag `hl_range` / `atr_norm` / `ADX` one session (`config.POST_SETTLEMENT_FEATURES`); Bitcoin (00:00 UTC bar, after the US close) keeps same-day values. Two new unit tests enforce the rule; the previous results are archived in `results/archive_sameday_macro_leak/`.
- **Results:** Silver's apparent 62.2 % directional accuracy / DM p = 0.002 disappears (served GRU 55.2 %, p = 0.12; DM p = 0.99). Final answer for all three assets: no model beats the random walk (served models 0.0–0.6 % above naive RMSE, DM p ≥ 0.71). Served models: CatBoost (Bitcoin), CatBoost (Gold), GRU (Silver). Bitcoin's numbers are bit-identical to the previous run.
- **Live sync** downloads into `data/raw_live/` and refreshes gold for Silver's `gold_return`; the frozen `data/raw/` is never touched.
- **Reproducibility:** `make preprocess` / `make collect-data` work directly (missing `sys.path` bootstrap); Keras phase A and phase B are now the same procedure (early stopping only); Sharpe annualised with 365 days for Bitcoin; the demo look-ahead test now perturbs every future row.
- Docs, README, viva notes, demo guide, both PDFs and the notebooks regenerated from the new results. Tests: 34 cases.

## [Improvement phase] — 2026-09-12 — more data, measured improvements, demonstration mode

- **Data:** full Yahoo Finance history 2018-01 → 2026-09-12 (2.9× more training data; test window extended to 2026-09-12). 3-year raw files and results archived (`data/raw_3y_backup/`, `results/archive_3y_final/`).
- **Features:** HAR realised volatilities (`rv_1d/5d/22d`), RiskMetrics EWMA volatility, 20-day momentum (Bitcoin 29 / Gold 28 / Silver 29 features).
- **Targets/tasks:** `build_dataset(asset, task=…)` with `return_1d` (served), `return_5d`, `vol_5d`, `vol_22d`; split defined by the dates the target covers (exact purge/embargo); targets standardised with training statistics.
- **Experiments** (`src/experiments/`): E1 data size, E2 feature-group ablation, E3 horizon, E4 volatility target — validation only; `before_after.py` compares the two systems on identical unseen days.
- **Evaluation:** prediction history with per-day error and direction hit; regime analysis (volatility terciles, trend, up/down days); volatility metrics (QLIKE, log-RMSE).
- **Dashboard:** *Predict a Day (unseen test)* — pick any test day, predict the next close from data up to that day, reveal the actual, error, hit/miss, highlighted on the actual-vs-predicted chart, all models on that day; *Prediction History* tab with MAE / hit-rate KPIs and CSV download; regime table and experiment tables on the Performance tab; `?asset=…&run=1` deep links.
- **Results (superseded — see the final-audit entry above):** this phase reported Gold −1.9 % and Silver −2.8 % return-RMSE on identical unseen days and a significant Silver result (62.2 % direction); the Silver figures were caused by the close-time leak fixed on 2026-09-13.
- Tests: 31 cases (task targets, embargo, standardisation round-trip, predict-for-date look-ahead check).

## [FYP release] — 2026-09-12 — Audit → fix → validated pipeline

Full list with rationale in `docs/FIX_PLAN.md`; original findings in `docs/AUDIT_SUMMARY.md`.

### Methodology (critical)
- **Fixed price reconstruction** — `reconstruct_price` used the previous day's *open* (column 0) instead of the *close*; every USD metric and dashboard forecast was wrong by one intraday move. Now takes the explicit previous close; regression test asserts exact reconstruction of the true target.
- **Removed synthetic weekend rows for Gold/Silver** — futures now keep their exchange calendar (30 % of rows were forward-filled zero-return fakes).
- **Stationary features only** — price levels (open/high/low/close/volume/EMA/BB/ATR/MACD) are chart-only; models get ratios (`price/EMA−1`, `%B`, `ATR/price`, `MACD/price`, HL range) plus returns, volatility, RSI/ADX/ROC, macro returns, sentiment. Added `gold_return` for Silver.
- **Walk-forward hyper-parameter tuning** (`src/training/tune_models.py`) on train+val only, replacing hyper-parameters that had been chosen on the test set in April.
- **Two-phase training for every model with a stopping rule**; CatBoost no longer early-stops on its own training data.
- **Honest evaluation** (`src/utils/metrics.py`, `src/evaluation/backtesting.py`): return-space RMSE/MAE/R², correct directional accuracy on non-flat days with binomial p-value, Diebold–Mariano test vs the random walk, strategy backtest vs buy-and-hold, train-vs-validation over-fitting gap. R² on price levels is no longer reported.
- **Model selection by validation only**; `data/models/model_status.json` is written by the evaluation script, not hand-edited.

### Models
- Unified registry (`src/models/registry.py`): Naive-Zero, Naive-Mean, ARIMA, Ridge, RandomForest, LightGBM, CatBoost, GRU, LSTM; stacked ensemble rebuilt with non-negative weights as a reported experiment.
- GRU/LSTM reduced to a single tunable layer (the old 2×100-unit stack over-fitted ~500 windows).
- Removed `advanced_models.py`, `baseline_models.py`, `catboost_model.py`, `model_lgbm.py`, `arima_model.py`, the three near-duplicate `train_final_*.py` scripts.

### System
- `src/inference/prediction.py`, `src/api/app.py`, `app/streamlit_app.py` rewritten around the served model; the previous versions crashed (LightGBM given 900 features, `float(dict)`) or silently failed (stale LSTM, missing Ensemble branch). The dashboard now shows horizon, model, held-out metrics, uncertainty band, disclaimer, a forecast-horizon chart and return-space validation plots.
- Live sync writes `<prefix>_live_features.csv` and never overwrites the frozen evaluation data.
- Data collection supports `DATA_START_DATE` (longer history).
- Results moved to `results/` (as `config.RESULTS_DIR` always said); pre-fix results archived in `results/archive_pre_fix/` with an explanation.
- Makefile, CI, `scripts/retrain.py` point at scripts that exist; new test suite (25 tests) covers look-ahead, target alignment, split integrity, reconstruction, metrics, model contract and inference sanity.
- Documentation: `docs/METHODOLOGY.md`, `docs/FEATURES.md`, `docs/LIMITATIONS.md`, `docs/RESULTS.md`, `docs/VIVA_QA.md`, `docs/REPORT_STRUCTURE.md`, `docs/FIX_PLAN.md`, `docs/AUDIT_SUMMARY.md`; README corrected (no PatchTST/TFT/Williams %R/CCI claims).

## [Week 1] — earlier correctness fixes (historical)
- Experiment scripts used `validation_data=(X_test, y_test)` — fixed at the time, scripts later deleted.
- Pinned dependencies, reproducibility seeds, `seq_len` consistency, `.gitignore` for `.env`.
