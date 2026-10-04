# Multi-Asset Next-Day Return Prediction — Bitcoin, Gold, Silver

*Final Year Project (BSCS) — a leakage-audited, walk-forward-validated comparison of statistical, tree-based and recurrent models against the random-walk baseline, with a live prediction dashboard and API.*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange) ![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.50-red) ![Tests](https://img.shields.io/badge/tests-46%20passing-brightgreen)

## What the project does

1. Downloads daily OHLCV for **BTC-USD**, **GC=F** (gold futures) and **SI=F** (silver futures) from 2018 plus macro series (DXY, WTI, 10-y yield, S&P 500, VIX) and the crypto Fear & Greed index — **complete bars only** (the running bar of the current day is dropped).
2. Engineers **stationary, backward-looking features** (returns and 20-day momentum, rolling/HAR/EWMA volatility, RSI/ADX/ROC, normalised MACD, EMA ratios, Bollinger %B/width, ATR/price, macro returns, sentiment) under a **close-time rule**: a feature may only use what is known when the asset's daily close is fixed — same-day macro values for Bitcoin (00:00 UTC bar), previous-day values for the metals (13:30 ET COMEX settlement) (the feature dictionary is Table 8 of `docs/FYP_Technical_Report.pdf`).
3. Predicts the **next-day log return** `y = ln(P_{t+1}/P_t)` and converts it to a price.
4. Compares **Naive (random walk), historical mean, ARIMA, Ridge, Random Forest, LightGBM, CatBoost, GRU, LSTM and a stacked ensemble** under **expanding-window walk-forward validation**, tunes hyper-parameters on validation only, and evaluates the untouched test set **once** with return-space metrics, directional accuracy with significance, a Diebold–Mariano test against the random walk, and a strategy backtest.
5. Serves one **combined forecast** — the equal-weight mean of all six trained models, so the user never picks a model — in a **Streamlit dashboard** (Run prediction with a **live scorecard of every model on the most recent days**, **Predict-a-Day demo on the unseen test period**, performance and methodology tabs) and a **FastAPI** endpoint, always alongside its held-out metrics, a **calibrated, conditional 68 % band** (`P_t·exp(r̂ ± k·σ_t)`, σ_t = EWMA volatility at the last complete bar — the one quantity this data does predict — and k ≈ 0.83–0.91 calibrated on training-period walk-forward errors) and a disclaimer.
6. Logs design experiments — data size, feature ablation, horizon, volatility target, and the final-audit candidates (ensemble membership, de-duplicated features, zero-drift forecast) — and a before/after comparison on identical unseen days (`results/experiments/`).

The central research question is honest: *does any model beat the random walk at a one-day horizon, and by how much?* Answer (`docs/FYP_Technical_Report.pdf` §14–15): **no — not for Bitcoin, Gold or Silver.** The served combined forecast's return-RMSE is within ±0.3 % of the random walk's on the unseen 2026 test window (Diebold–Mariano p = 0.46–0.97), directional accuracy 47–54 % (none significant); no single model does better. The same holds on **all ≈ 6 years of walk-forward out-of-fold predictions pooled together** (1,584–2,328 days per asset, ten times the test window): Combined vs random walk −0.02 % / −0.21 % / −0.06 % RMSE, DM p = 0.87 / 0.09 / 0.69, and the metals' 52–53 % direction hit-rates are below the "always UP" drift forecast's 53.5–55.8 % (`results/walkforward_pooled.csv`). An intermediate version reported 62 % for Silver; the final audit traced it to a close-time leak (same-day macro features against a 13:30 ET futures settlement), fixed it and re-ran everything (`docs/FYP_Technical_Report.pdf` §11 and §15.2).

## Results at a glance

Served (Combined) forecast vs the random walk on the untouched test set (`results/final_test_results.csv`, evaluated once; MAE/RMSE/MAPE in USD, RMSE and R² of the log return, DA = direction correct on non-flat days with its one-sided binomial p, "UP calls" = share of days the forecast said UP, DM = Diebold–Mariano p vs the random walk):

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

Reading: every error metric is within ±0.4 % of the random walk (no difference is significant), R² of the return is ≈ 0, and no directional hit-rate is significantly above 50 %. The predicted returns have a standard deviation of ~0.1 % against ~2 % for actual returns — the models have learned that the next-day move is essentially unpredictable and stay near "no change". For Gold and Silver the forecast says UP on 91–97 % of days, so its direction call is the assets' average drift rather than a timing signal. Full tables (every model, walk-forward folds, regimes, experiments) in **`results/FINAL_RESULTS.md`**; interpretation in **`docs/FYP_Technical_Report.pdf`** §14–15; figures in `results/figures/`.

**Final report:** `docs/FYP_Technical_Report.pdf` — the complete technical report (data, features, models, evaluation, dashboard, limitations); every table in it was generated from the result files (the "UP calls" column above was added in the final audit and is not in the PDF's tables).

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
│   │   └── before_after.py      3-year system vs improved system on identical unseen days
│   ├── evaluation/
│   │   ├── cross_validation.py  expanding-window folds
│   │   ├── backtesting.py       ONE evaluation on the test set (every model + the Combined forecast), CV comparison, pooled walk-forward tests, band calibration, results tables
│   │   └── plots.py             report figures
│   ├── inference/prediction.py  next-day prediction (Combined = mean of all trained models, calibrated 68 % EWMA-volatility band), predict-for-date demo
│   ├── api/app.py               FastAPI
│   └── utils/                   metrics (DM test, directional accuracy, backtest), reconstruction, seeds, logging
├── tests/                     46 test cases: look-ahead (crypto + futures), close-time alignment, complete-bar rule, target alignment, split, reconstruction, metrics, combined inference, conditional band, failure modes
├── results/                   cv_results.csv · final_test_results.csv · walkforward_pooled.csv · band_calibration.csv · regime_analysis.csv · FINAL_RESULTS.md · tuning/ · figures/ · predictions/ · experiments/ · archive_3y_final/ · archive_sameday_macro_leak/
├── docs/                      FYP_Technical_Report.pdf — the complete technical report
├── notebooks/                 companion notebooks (load and display the script outputs)
└── data/                      raw/ processed/ models/  (git-ignored except model_status.json)
```

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

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

`make pipeline` runs everything from `preprocess` to `test`; `python scripts/retrain.py` additionally re-downloads data. `make collect-data` fetches from `config.DATA_START_DATE` (2018-01-01); all reported results use the 2018-01 → 2026-09-12 download (the earlier 3-year files are kept in `data/raw_3y_backup/`). `data/` is git-ignored: on a fresh clone run `make collect-data preprocess train stack evaluate` (≈ 15 min without tuning, using the committed `results/tuning/best_params.json`) before `make serve`.

## Methodology in one paragraph

Chronological split by the dates the target covers (train ≤ 2025-09-10, validation ≤ 2026-02-17, test = rest, exact embargo for multi-day targets), no shuffling; scaler fitted on train only; only complete daily bars; Gold/Silver keep their exchange calendar (no synthetic weekend rows); every feature at day *t* uses only information known when the day-*t* close is fixed — same-day macro data for Bitcoin, previous-day macro data and previous-session High/Low for the metals, whose Yahoo close is the 13:30 ET settlement (tested); the target is the next row's log return and never appears in the input window (tested); hyper-parameters are chosen by 4-fold expanding-window walk-forward validation inside train+val; the served forecast is the equal-weight combination of the six trained models (no fitted weights, so nothing is selected on the test set), evaluated on the folds and once on the test set like every single model; the test set is read by exactly one script. Full detail: `docs/FYP_Technical_Report.pdf` §7–13.

## Limitations (stated, not hidden)

- **No point-forecast skill.** Next-day returns of these assets are statistically indistinguishable from a random walk with this data; the system is a rigorous *test* of that question, not a trading tool.
- **The test window is short** (143–207 days) and was inspected during development (e.g. when the close-time leak was found and when the Combined forecast and the band were introduced), so it is not a pristine one-shot holdout. The pooled walk-forward record (≈ 6 years, train+val only) is the larger-sample evidence, and it agrees.
- **Metals' "direction" is drift.** Models trained on 2018–2025 learn gold's and silver's rising average and say UP on 74–97 % of days; their hit-rate is the share of up days, not timing.
- **Data vendor.** Yahoo's continuous futures (GC=F / SI=F) follow the front-month contract and can be revised after a roll; the frozen dataset behind every reported number is unchanged.

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
