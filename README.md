# Multi-Asset Next-Day Return Prediction — Bitcoin, Gold, Silver

*Final Year Project (BSCS) — a leakage-audited, walk-forward-validated comparison of statistical, tree-based and recurrent models against the random-walk baseline, with a live prediction dashboard and API.*

![Python](https://img.shields.io/badge/Python-3.10–3.12-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange) ![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.50-red) ![Tests](https://img.shields.io/badge/tests-55%20passing-brightgreen)

## What the project does

1. Downloads daily OHLCV for **BTC-USD**, **GC=F** (gold futures) and **SI=F** (silver futures) from 2018 plus macro series (DXY, WTI, 10-y yield, S&P 500, VIX) and the crypto Fear & Greed index — **complete bars only** (the running bar of the current day is dropped).
2. Engineers **stationary, backward-looking features** (returns and 20-day momentum, rolling/HAR/EWMA volatility, RSI/ADX/ROC, normalised MACD, EMA ratios, Bollinger %B/width, ATR/price, macro returns, sentiment) under a **close-time rule**: a feature may only use what is known when the asset's daily close is fixed — same-day macro values for Bitcoin (00:00 UTC bar), previous-day values for the metals (13:30 ET COMEX settlement) (the feature dictionary is Table 8 of `docs/FYP_Technical_Report.pdf`).
3. Predicts the **next-day log return** `y = ln(P_{t+1}/P_t)` and converts it to a price.
4. Compares **Naive (random walk), historical-mean drift, ARIMA, two moving-average rules (5-day average price, 20-day mean return), Ridge, Random Forest, LightGBM, CatBoost, GRU, LSTM and a stacked ensemble** on identical days under **expanding-window walk-forward validation**, tunes hyper-parameters on validation only, and evaluates the held-out test set in a single evaluation script with return-space metrics, directional accuracy with significance, a Diebold–Mariano test against the random walk, and a strategy backtest.
5. Serves one **combined forecast** — the equal-weight mean of all six trained models, so the user never picks a model — in a **Streamlit dashboard** (Run prediction with a **live scorecard of every model on the most recent days**, **Predict-a-Day demo on the unseen test period**, performance and methodology tabs) and a **FastAPI** endpoint, always alongside its held-out metrics, a **calibrated, conditional 68 % band** (`P_t·exp(r̂ ± k·σ_t)`, σ_t = EWMA volatility at the last complete bar — the one quantity this data does predict — and k ≈ 0.83–0.91 calibrated on training-period walk-forward errors) and a disclaimer.
6. Logs design experiments — data size, feature ablation, horizon, volatility target, the final-audit candidates (ensemble membership, de-duplicated features, zero-drift forecast) and the submission-audit candidates (volatility-scaled target, robust Huber loss) — and a before/after comparison on identical unseen days (`results/experiments/`).

The central research question is honest: *does any model beat the random walk at a one-day horizon, and by how much?* Answer (`docs/FYP_Technical_Report.pdf` §14–15): **no — not for Bitcoin, Gold or Silver.** The served combined forecast's return-RMSE is within ±0.3 % of the random walk's on the unseen 2026 test window (Diebold–Mariano p = 0.46–0.97), directional accuracy 47–54 % (none significant); no single model does better. The same holds on **all ≈ 6 years of walk-forward out-of-fold predictions pooled together** (1,584–2,328 days per asset, ten times the test window): Combined vs random walk −0.02 % / −0.21 % / −0.06 % RMSE, DM p = 0.87 / 0.09 / 0.69, and the metals' 52–53 % direction hit-rates are below the "always UP" drift forecast's 53.5–55.8 % (`results/walkforward_pooled.csv`). An intermediate version reported 62 % for Silver; the final audit traced it to a close-time leak (same-day macro features against a 13:30 ET futures settlement), fixed it and re-ran everything (`docs/FYP_Technical_Report.pdf` §11 and §15.2).

## Results at a glance

Served (Combined) forecast vs the random walk on the held-out test set (`results/final_test_results.csv`, one evaluation run; MAE/RMSE/MAPE in USD, RMSE and R² of the log return, DA = direction correct on non-flat days with its one-sided binomial p, "UP calls" = share of days the forecast said UP, DM = Diebold–Mariano p vs the random walk):

| Asset | Test days | Forecast | MAE ($) | RMSE ($) | MAPE | RMSE (ret) | R² (ret) | DA (p) | UP calls / up days | DM p |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Bitcoin | 207 | **Combined** | 1,063.94 | 1,447.37 | 1.52 % | 0.02072 | +0.006 | 46.9 % (0.83) | 37 % / 49 % | 0.49 |
| | | Random walk | 1,066.42 | 1,451.73 | 1.52 % | 0.02079 | −0.001 | — | — | — |
| Gold | 143 | **Combined** | 58.20 | 75.73 | 1.29 % | 0.01671 | −0.001 | 50.3 % (0.50) | 97 % / 50 % | 0.97 |
| | | Random walk | 58.26 | 75.76 | 1.29 % | 0.01671 | −0.002 | — | — | — |
| Silver | 143 | **Combined** | 1.78 | 2.37 | 2.47 % | 0.03180 | +0.005 | 53.8 % (0.20) | 91 % / 52 % | 0.46 |
| | | Random walk | 1.79 | 2.37 | 2.49 % | 0.03189 | −0.001 | — | — | — |

Long-run evidence (`results/walkforward_pooled.csv`, every walk-forward out-of-fold day pooled, train+val only):

| Asset | Out-of-fold days | Combined RMSE vs random walk | DM p | Direction correct (p vs 50 %) | "Always UP" drift forecast |
|---|--:|--:|--:|--:|--:|
| Bitcoin | 2,328 (2019-10 → 2026-02) | −0.018 % | 0.87 | 50.8 % (0.23) | 48.8 % |
| Gold | 1,584 (2019-10 → 2026-02) | −0.213 % | 0.09 | 52.5 % (0.02) | 55.8 % |
| Silver | 1,584 (2019-10 → 2026-02) | −0.058 % | 0.69 | 52.7 % (0.02) | 53.5 % |

68 % uncertainty band (`results/band_calibration.csv`; k fitted on training-split walk-forward errors, so validation and test are out-of-sample) — share of unseen test days whose actual close fell inside the band:

| Asset | k | Served band (k·σ_t) | Previous ±1σ_t band | Fixed-width band |
|---|--:|--:|--:|--:|
| Bitcoin | 0.83 | 65.7 % | 73.9 % | 79.2 % |
| Gold | 0.91 | 69.2 % | 75.5 % | 42.0 % |
| Silver | 0.87 | 69.2 % | 75.5 % | 44.8 % |

(On the shorter validation window the calibrated band covered 60.6 / 66.1 / 54.1 % vs 75.0 / 72.5 / 57.8 % for ±1σ — better for Gold, worse for Bitcoin and Silver; Silver's late-2025 volatility jump makes every EWMA band under-cover there.)

Every forecast against every baseline, same unseen test days (RMSE of the next-day return relative to the random walk; negative = smaller error; `results/final_test_results.csv`):

| Asset | Combined (served) | Best single model by CV | Drift (training mean) | ARIMA | MA-5 (5-day average price) | Mean-20d (20-day mean return) |
|---|--:|--:|--:|--:|--:|--:|
| Bitcoin | −0.34 % | CatBoost +0.13 % | −0.05 % | +0.56 % | +57.0 % | +2.68 % |
| Gold | −0.04 % | CatBoost +0.62 % | +0.26 % | +0.62 % | +52.3 % | +4.17 % |
| Silver | −0.29 % | GRU −0.01 % | +0.10 % | +1.14 % | +55.5 % | +5.73 % |

The two moving-average rules are the "naive" benchmarks often used in forecasting projects — and they are 2.7–57 % *worse* than the random walk (DM p < 0.05 everywhere except Bitcoin's Mean-20d, p = 0.16). Beating a moving average says nothing; the random walk ("tomorrow = today") is the benchmark that matters, and nothing here beats it significantly.

Reading: every error metric is within ±0.4 % of the random walk (no difference is significant), R² of the return is ≈ 0, and no directional hit-rate is significantly above 50 %. The predicted returns have a standard deviation of ~0.1 % against ~2 % for actual returns — the models have learned that the next-day move is essentially unpredictable and stay near "no change". For Gold and Silver the forecast says UP on 91–97 % of days, so its direction call is the assets' average drift rather than a timing signal. Full tables (every model, walk-forward folds, regimes, experiments) in **`results/FINAL_RESULTS.md`**; interpretation in **`docs/FYP_Technical_Report.pdf`** §14–15; figures in `results/figures/`.

**Fresh hold-out (2026-11-13):** the frozen models, unchanged, scored on the 61 (Bitcoin) / 44 (Gold, Silver) daily closes published *after* the frozen dataset — days no version of the project had seen (`results/experiments/fresh_holdout.md`, `make fresh-holdout`). Same answer: Combined vs random walk −0.51 % / +1.34 % / +0.99 % RMSE, DM p = 0.41 / 0.29 / 0.17 — no skill.

**Final report:** `docs/FYP_Technical_Report.pdf` — the complete technical report (data, features, models, evaluation, dashboard, limitations); every table in it is generated from the result files by `docs/build_technical_report.py` (`make report`). Edition of 13 November 2026: it documents the final system — five baselines, the pooled walk-forward evidence, the calibrated band, the live scorecard and the fresh hold-out.

## Project structure

```
├── config.py                  paths, frozen split dates, target, feature policy, default hyper-parameters
├── app/streamlit_app.py       dashboard (forecast · predict-a-day demo · performance · methodology)
├── src/
│   ├── data/
│   │   ├── data_collection.py   Yahoo Finance OHLCV (complete bars only)
│   │   ├── external_data.py     macro + Fear & Greed
│   │   ├── market_calendar.py   complete-bar rule (UTC day for crypto, 17:15 ET for futures / US series), next trading day
│   │   ├── preprocessing.py     cleaning, features, chronological split, scaler, build_dataset()
│   │   ├── eda.py               ADF tests, ACF/PACF, return distributions
│   │   └── sync_live_data.py    live refresh for demos into data/raw_live/ (never overwrites the frozen data)
│   ├── models/
│   │   ├── registry.py          uniform interface for all models (two-phase fitting)
│   │   ├── model_gru.py / model_lstm.py
│   │   └── ensemble_model.py    stacked ensemble (experiment)
│   ├── training/
│   │   ├── tune_models.py       walk-forward hyper-parameter search (train+val only, per task)
│   │   └── train_models.py      final training, train/val metrics, loss curves, feature importance
│   ├── experiments/
│   │   ├── run_experiments.py   E1 data size · E2 feature ablation · E3 horizon · E4 volatility (validation only)
│   │   ├── audit_improvements.py E5 final-audit candidates: membership · de-duplicated features · zero drift (validation only)
│   │   ├── final_candidates.py  E6 submission-audit candidates: volatility-scaled target · Huber loss (validation only)
│   │   ├── before_after.py      3-year system vs improved system on identical unseen days
│   │   └── fresh_holdout.py     the frozen models on the days published AFTER the frozen dataset (never seen by any version)
│   ├── evaluation/
│   │   ├── cross_validation.py  expanding-window folds
│   │   ├── backtesting.py       ONE evaluation on the test set (every model + the Combined forecast), CV comparison, pooled walk-forward tests, band calibration, results tables
│   │   └── plots.py             report figures
│   ├── inference/prediction.py  next-day prediction (Combined = mean of all trained models, calibrated 68 % EWMA-volatility band), predict-for-date demo
│   ├── api/app.py               FastAPI
│   └── utils/                   metrics (DM test, directional accuracy, backtest), reconstruction, seeds, logging
├── tests/                     55 test cases: look-ahead (crypto + futures), close-time alignment, complete-bar rule, target alignment, split, reconstruction, metrics, combined inference, conditional band, moving-average baselines, all-or-nothing live sync, failure modes
├── results/                   cv_results.csv · final_test_results.csv · walkforward_pooled.csv · band_calibration.csv · regime_analysis.csv · FINAL_RESULTS.md · tuning/ · figures/ · predictions/ · experiments/ · archive_3y_final/ · archive_sameday_macro_leak/
├── docs/                      FYP_Technical_Report.pdf — the complete technical report
├── notebooks/                 companion notebooks (load and display the script outputs)
└── data/                      raw/ (frozen dataset, versioned) · models/ (trained models + scalers, versioned) · processed/ (rebuilt by `make preprocess`) · raw_live/ (live downloads, git-ignored)
```

## Quick start

**Python 3.10–3.12 is required** — TensorFlow 2.16.2 has no wheels for Python 3.13+, so `pip install` fails on a newer system Python (use `python3.11`, or `uv venv -p 3.11 venv`). `requirements-lock.txt` is the exact environment the tests and the evaluation were verified with (Linux, Python 3.11).

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements-lock.txt      # or requirements.txt for the loose pins

make preprocess     # features + frozen chronological split + scaler (from data/raw)
make eda            # stationarity tests, ACF/PACF, distributions
make tune           # walk-forward hyper-parameter search  (~30 min; GRU/LSTM dominate)
make train          # two-phase training of all models
make stack          # stacked-ensemble experiment
make evaluate       # the single test-set evaluation + model selection
make figures        # report figures
make experiments    # design experiments + before/after comparison (validation only)
make test           # test suite
make predict        # print today's combined next-day forecast for every asset
make serve          # streamlit dashboard  →  http://localhost:8501
make api            # uvicorn API          →  http://localhost:8000/docs
```

**Fresh clone → working dashboard in two commands.** The frozen dataset (`data/raw/`, the 2018-01 → 2026-09-12 download behind every reported number) and the trained models (`data/models/`) are versioned, so after `pip install -r requirements.txt`:

```bash
make preprocess     # rebuilds data/processed/ from data/raw (byte-identical to the files the results were computed on)
make serve          # dashboard; "Run prediction" downloads the latest complete bars into data/raw_live/
```

`make evaluate` then reproduces every result file to ≤ 1e-15 on the original machine (verified on 2026-10-10). Re-verified on Linux / Python 3.11 on 2026-11-13: every test-set metric of the served Combined forecast and of every trained model is identical; the only differences are numerical noise from refitting across platforms — Silver's ARIMA baseline (2·10⁻⁶ in the RMSE) and the GRU/LSTM walk-forward refits (≤ 0.2 % relative in a few fold statistics). `make pipeline` runs everything from `preprocess` to `test` (re-tuning takes ~30 min). **Do not run `make collect-data` / `scripts/retrain.py` to reproduce the report:** they re-download from Yahoo, which revises history and rolls futures contracts, and they overwrite `data/raw/` (git keeps the frozen copy: `git checkout data/raw`). The earlier 3-year files are kept in `data/raw_3y_backup/`.

## Methodology in one paragraph

Chronological split by the dates the target covers (train ≤ 2025-09-10, validation ≤ 2026-02-17, test = rest, exact embargo for multi-day targets), no shuffling; scaler fitted on train only; only complete daily bars; Gold/Silver keep their exchange calendar (no synthetic weekend rows); every feature at day *t* uses only information known when the day-*t* close is fixed — same-day macro data for Bitcoin, previous-day macro data and previous-session High/Low for the metals, whose Yahoo close is the 13:30 ET settlement (tested); the target is the next row's log return and never appears in the input window (tested); hyper-parameters are chosen by 4-fold expanding-window walk-forward validation inside train+val; the served forecast is the equal-weight combination of the six trained models (no fitted weights, so nothing is selected on the test set), evaluated on the folds and once on the test set like every single model; the test set is read by exactly one script. Full detail: `docs/FYP_Technical_Report.pdf` §7–13.

## Limitations (stated, not hidden)

- **No point-forecast skill.** Next-day returns of these assets are statistically indistinguishable from a random walk with this data; the system is a rigorous *test* of that question, not a trading tool.
- **The test window is short** (143–207 days) and was inspected during development (e.g. when the close-time leak was found and when the Combined forecast and the band were introduced), so it is not a pristine one-shot holdout. The days published after the frozen dataset are (`results/experiments/fresh_holdout.md`: same answer, no skill). The pooled walk-forward record (≈ 6 years, train+val only) is the larger-sample evidence, and it agrees.
- **Metals' "direction" is drift.** Models trained on 2018–2025 learn gold's and silver's rising average and say UP on 74–97 % of days; their hit-rate is the share of up days, not timing.
- **Data vendor.** Yahoo's continuous futures (GC=F / SI=F) follow the front-month contract and can be revised after a roll (e.g. two flat Gold bars on 2026-01-27/28 with High = Low = Close). More generally the free futures series contains many degenerate bars with High = Low — 130 Gold days and 384 Silver days (18 %) of 2,186, mostly 2018–2021, 36 / 138 of them with zero volume — which flatten the range-based features (`hl_range`, `atr_norm`, ADX) on those days; the ablation (E2) shows these features carry no predictive value either way; the frozen dataset behind every reported number is unchanged. Genuine extreme days are kept, not removed: Bitcoin −37 % on 2020-03-12, WTI at −$37.63 on 2020-04-20, silver −31 % and gold −11 % on 2026-01-30 (inside the validation window).
- **Walk-forward folds share the training-split scaler.** The MinMax scaler and the target's mean/std are fitted once on the whole training split, so the early walk-forward folds are scaled with statistics that include later training data. This touches only model selection, the pooled walk-forward evidence and the band's k (and only the scale-sensitive Ridge/GRU/LSTM — tree models are invariant to it); the test evaluation is unaffected. Any bias favours the models, which still do not beat the random walk.
- **Phase-A validation metrics are mildly optimistic.** `results/train_val_metrics.csv` / `overfitting_gap.png` score LightGBM, CatBoost, GRU and LSTM on the same validation set they early-stop on. Model selection never uses them (it uses the walk-forward folds), and the test / walk-forward / fresh hold-out results are unaffected.
- **Small alignment simplifications (documented, not leaks of any size that matters).** Silver uses gold's same-day settlement (13:30 ET) although silver settles at 13:25 ET — a 5-minute overlap at a daily horizon. On weekends Bitcoin sees Friday's S&P 500 / VIX return carried forward (a stale value, not future information). WTI's negative close on 2020-04-20 makes two oil log returns undefined; they are set to 0.
- **Feature names.** `log_return` is day t's own return (known at the close of t); `return_1d`, `return_2d`, … are the returns of 1, 2, … days *earlier*; `rv_1d` = |`log_return`| duplicates it in magnitude.
- **Many tests.** The tables report dozens of significance tests; a few p-values below 0.05 are expected by chance (e.g. the MA-5 rule's 52.1 % direction hit-rate on Bitcoin's walk-forward days, p = 0.022, which is 52.7 %, p = 0.24, on the test window). Nothing survives a multiple-testing correction.
- **Live mode depends on Yahoo.** A live refresh is all-or-nothing: if any series a forecast needs is not downloaded, the previous live file (or the frozen dataset) stays in use and the API reports `data_is_current: false`.

## API

```
GET /health
GET /models                       served forecast (Combined), its members, the CV-selected single model, held-out metrics per asset
GET /predict/{bitcoin|gold|silver}[?model=GRU]     default = the combined forecast with every member's own prediction
```

## Disclaimer

Research prototype for an academic project. Next-day financial returns are close to unpredictable; the system reports its own held-out performance next to every forecast. Not financial advice.

---
<div align="center"><i>University of Lahore — BSCS — teamlocalhost</i></div>
