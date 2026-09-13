# Multi-Asset Next-Day Return Prediction — Bitcoin, Gold, Silver

*Final Year Project (BSCS) — a leakage-audited, walk-forward-validated comparison of statistical, tree-based and recurrent models against the random-walk baseline, with a live prediction dashboard and API.*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange) ![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.50-red) ![Tests](https://img.shields.io/badge/tests-34%20passing-brightgreen)

## What the project does

1. Downloads daily OHLCV for **BTC-USD**, **GC=F** (gold futures) and **SI=F** (silver futures) from 2018 plus macro series (DXY, WTI, 10-y yield, S&P 500, VIX) and the crypto Fear & Greed index.
2. Engineers **stationary, backward-looking features** (returns and 20-day momentum, rolling/HAR/EWMA volatility, RSI/ADX/ROC, normalised MACD, EMA ratios, Bollinger %B/width, ATR/price, macro returns, sentiment) under a **close-time rule**: a feature may only use what is known when the asset's daily close is fixed — same-day macro values for Bitcoin (00:00 UTC bar), previous-day values for the metals (13:30 ET COMEX settlement) — `docs/FEATURES.md`.
3. Predicts the **next-day log return** `y = ln(P_{t+1}/P_t)` and converts it to a price.
4. Compares **Naive (random walk), historical mean, ARIMA, Ridge, Random Forest, LightGBM, CatBoost, GRU, LSTM and a stacked ensemble** under **expanding-window walk-forward validation**, tunes hyper-parameters on validation only, and evaluates the untouched test set **once** with return-space metrics, directional accuracy with significance, a Diebold–Mariano test against the random walk, and a strategy backtest.
5. Serves the validation-selected model per asset in a **Streamlit dashboard** (live forecast, **Predict-a-Day demo on the unseen test period**, prediction history, performance and methodology tabs) and a **FastAPI** endpoint, always alongside its held-out metrics, an uncertainty band and a disclaimer.
6. Logs design experiments — data size, feature ablation, horizon, volatility target — and a before/after comparison on identical unseen days (`results/experiments/`).

The central research question is honest: *does any model beat the random walk at a one-day horizon, and by how much?* Answer (`docs/RESULTS.md`): **no — not for Bitcoin, Gold or Silver.** The served models' return-RMSE is 0.0–0.6 % above the random walk's on the unseen 2026 test window (Diebold–Mariano p ≥ 0.71), directional accuracy 47–55 % (none significant). An intermediate version reported 62 % for Silver; the final audit traced it to a close-time leak (same-day macro features against a 13:30 ET futures settlement), fixed it and re-ran everything (`docs/RESULTS.md §2`).

## Results at a glance

See **`docs/RESULTS.md`** (interpretation) and **`results/FINAL_RESULTS.md`** (auto-generated tables). Figures are in `results/figures/`.

**Final report:** `docs/FYP_Final_Report.pdf` and `docs/Executive_Summary.pdf`, both generated from the result files by `python docs/build_report.py`. **Demo script:** `docs/DEMO_GUIDE.md`.

## Project structure

```
├── config.py                  paths, frozen split dates, target, feature policy, default hyper-parameters
├── app/streamlit_app.py       dashboard (forecast · predict-a-day demo · history · performance · methodology)
├── src/
│   ├── data/
│   │   ├── data_collection.py   Yahoo Finance OHLCV
│   │   ├── external_data.py     macro + Fear & Greed
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
│   │   └── before_after.py      3-year system vs improved system on identical unseen days
│   ├── evaluation/
│   │   ├── cross_validation.py  expanding-window folds
│   │   ├── backtesting.py       ONE evaluation on the test set, model selection, results tables
│   │   └── plots.py             report figures
│   ├── inference/prediction.py  next-day prediction, predict-for-date demo, prediction history
│   ├── api/app.py               FastAPI
│   └── utils/                   metrics (DM test, directional accuracy, backtest), reconstruction, seeds, logging
├── tests/                     34 test cases: look-ahead (crypto + futures), close-time alignment, target alignment, split, reconstruction, metrics, inference
├── results/                   cv_results.csv · final_test_results.csv · regime_analysis.csv · FINAL_RESULTS.md · tuning/ · figures/ · predictions/ · experiments/ · archive_3y_final/ · archive_sameday_macro_leak/
├── docs/                      METHODOLOGY · FEATURES · RESULTS · LIMITATIONS · VIVA_QA · DEMO_GUIDE · REPORT_STRUCTURE · FIX_PLAN · AUDIT_SUMMARY
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
make serve          # streamlit dashboard  →  http://localhost:8501
make api            # uvicorn API          →  http://localhost:8000/docs
```

`make pipeline` runs everything from `preprocess` to `test`; `python scripts/retrain.py` additionally re-downloads data. `make collect-data` fetches from `config.DATA_START_DATE` (2018-01-01); all reported results use the 2018-01 → 2026-09 download (the earlier 3-year files are kept in `data/raw_3y_backup/`).

## Methodology in one paragraph

Chronological split by the dates the target covers (train ≤ 2025-09-10, validation ≤ 2026-02-17, test = rest, exact embargo for multi-day targets), no shuffling; scaler fitted on train only; Gold/Silver keep their exchange calendar (no synthetic weekend rows); every feature at day *t* uses only information known when the day-*t* close is fixed — same-day macro data for Bitcoin, previous-day macro data and previous-session High/Low for the metals, whose Yahoo close is the 13:30 ET settlement (tested); the target is the next row's log return and never appears in the input window (tested); hyper-parameters and the served model are chosen by 4-fold expanding-window walk-forward validation inside train+val; the test set is read by exactly one script. Full detail: `docs/METHODOLOGY.md`.

## API

```
GET /health
GET /models                       served model, selection rule, held-out metrics per asset
GET /predict/{bitcoin|gold|silver}[?model=GRU]
```

## Disclaimer

Research prototype for an academic project. Next-day financial returns are close to unpredictable; the system reports its own held-out performance next to every forecast. Not financial advice.

---
<div align="center"><i>University of Lahore — BSCS — teamlocalhost</i></div>
