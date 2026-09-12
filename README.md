# Multi-Asset Next-Day Return Prediction — Bitcoin, Gold, Silver

*Final Year Project (BSCS) — a leakage-audited, walk-forward-validated comparison of statistical, tree-based and recurrent models against the random-walk baseline, with a live prediction dashboard and API.*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange) ![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.50-red) ![Tests](https://img.shields.io/badge/tests-25%20passing-brightgreen)

## What the project does

1. Downloads daily OHLCV for **BTC-USD**, **GC=F** (gold futures) and **SI=F** (silver futures) plus macro series (DXY, WTI, 10-y yield, S&P 500, VIX) and the crypto Fear & Greed index.
2. Engineers **stationary, backward-looking features** (returns, volatility, RSI/ADX/ROC, normalised MACD, EMA ratios, Bollinger %B/width, ATR/price, macro returns, sentiment) — `docs/FEATURES.md`.
3. Predicts the **next-day log return** `y = ln(P_{t+1}/P_t)` and converts it to a price.
4. Compares **Naive (random walk), historical mean, ARIMA, Ridge, Random Forest, LightGBM, CatBoost, GRU, LSTM and a stacked ensemble** under **expanding-window walk-forward validation**, tunes hyper-parameters on validation only, and evaluates the untouched test set **once** with return-space metrics, directional accuracy with significance, a Diebold–Mariano test against the random walk, and a strategy backtest.
5. Serves the validation-selected model per asset in a **Streamlit dashboard** and a **FastAPI** endpoint, always alongside its held-out metrics, an uncertainty band and a disclaimer.

The central research question is honest: *does any model beat the random walk at a one-day horizon, and by how much?* See `docs/RESULTS.md` for the answer with real numbers.

## Results at a glance

See **`docs/RESULTS.md`** (interpretation) and **`results/FINAL_RESULTS.md`** (auto-generated tables). Figures are in `results/figures/`.

## Project structure

```
├── config.py                  paths, frozen split dates, target, feature policy, default hyper-parameters
├── app/streamlit_app.py       dashboard (forecast + indicators · performance · methodology)
├── src/
│   ├── data/
│   │   ├── data_collection.py   Yahoo Finance OHLCV
│   │   ├── external_data.py     macro + Fear & Greed
│   │   ├── preprocessing.py     cleaning, features, chronological split, scaler, build_dataset()
│   │   ├── eda.py               ADF tests, ACF/PACF, return distributions
│   │   └── sync_live_data.py    live refresh for demos (never overwrites the frozen data)
│   ├── models/
│   │   ├── registry.py          uniform interface for all models (two-phase fitting)
│   │   ├── model_gru.py / model_lstm.py
│   │   └── ensemble_model.py    stacked ensemble (experiment)
│   ├── training/
│   │   ├── tune_models.py       walk-forward hyper-parameter search (train+val only)
│   │   └── train_models.py      final training, train/val metrics, loss curves, feature importance
│   ├── evaluation/
│   │   ├── cross_validation.py  expanding-window folds
│   │   ├── backtesting.py       ONE evaluation on the test set, model selection, results tables
│   │   └── plots.py             report figures
│   ├── inference/prediction.py  next-day prediction with context
│   ├── api/app.py               FastAPI
│   └── utils/                   metrics (DM test, directional accuracy, backtest), reconstruction, seeds, logging
├── tests/                     25 tests: look-ahead, target alignment, split, reconstruction, metrics, inference
├── results/                   cv_results.csv · final_test_results.csv · FINAL_RESULTS.md · tuning/ · figures/ · predictions/
├── docs/                      METHODOLOGY · FEATURES · RESULTS · LIMITATIONS · VIVA_QA · REPORT_STRUCTURE · FIX_PLAN · AUDIT_SUMMARY
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
make test           # test suite
make serve          # streamlit dashboard  →  http://localhost:8501
make api            # uvicorn API          →  http://localhost:8000/docs
```

`make pipeline` runs everything from `preprocess` to `test`; `python scripts/retrain.py` additionally re-downloads data. `make collect-data` fetches from `config.DATA_START_DATE` (2018) — Yahoo Finance rate-limits aggressively, so the repository ships with the 2023-07 → 2026-07 data that all reported results use.

## Methodology in one paragraph

Chronological split by date (train ≤ 2025-09-10, validation ≤ 2026-02-17, test = rest), no shuffling; scaler fitted on train only; Gold/Silver keep their exchange calendar (no synthetic weekend rows); every feature at day *t* uses only data ≤ *t* (tested); the target is the next row's log return and never appears in the input window (tested); hyper-parameters and the served model are chosen by 4-fold expanding-window walk-forward validation inside train+val; the test set is read by exactly one script. Full detail: `docs/METHODOLOGY.md`.

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
