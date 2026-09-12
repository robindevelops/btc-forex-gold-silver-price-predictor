# Source Code Organization (`src/`)

| Package | File | Role |
|---|---|---|
| `data/` | `data_collection.py` | Yahoo Finance OHLCV for BTC-USD, GC=F, SI=F (`--start` for longer history) |
| | `external_data.py` | DXY, WTI, 10-y yield, S&P 500, VIX, Fear & Greed |
| | `preprocessing.py` | `DataCleaner` (cleaning, calendar, features), frozen chronological split, train-only scaler, `create_sequences`, **`build_dataset()`** — the single loader used everywhere |
| | `eda.py` | ADF tests, ACF/PACF, return distributions, volatility clustering |
| | `sync_live_data.py` | live refresh into `<prefix>_live_features.csv` (never overwrites frozen data) |
| `models/` | `registry.py` | uniform interface + factory for Naive-Zero, Naive-Mean, ARIMA, Ridge, RandomForest, LightGBM, CatBoost, GRU, LSTM; two-phase fitting; save/load |
| | `model_gru.py`, `model_lstm.py` | Keras builders |
| | `ensemble_model.py` | stacked ensemble on out-of-fold predictions (experiment) |
| `training/` | `tune_models.py` | walk-forward grid search on train+val → `results/tuning/best_params.json` |
| | `train_models.py` | phase A (train→val metrics, loss curves, importance) + phase B (deployed model on train+val) |
| `evaluation/` | `cross_validation.py` | expanding-window folds, `cv_evaluate()` |
| | `backtesting.py` | **the only reader of the test split**: final tables, model selection, `model_status.json` |
| | `plots.py` | report figures |
| `inference/` | `prediction.py` | `predict_next_day()`, `predict_all_models()` with held-out metrics and uncertainty band |
| `api/` | `app.py` | FastAPI |
| `utils/` | `metrics.py` | RMSE/MAE/MAPE/R², directional accuracy + binomial test, Diebold–Mariano, strategy backtest |
| | `inverse_transform.py` | `P̂ = P_t · exp(r̂)` |
| | `reproducibility.py`, `logging_config.py` | seeds, logging |

Data-flow: `data/raw → preprocessing → build_dataset → {tune, train, stack} → backtesting → results/ → {streamlit, api}`.
