"""
Central configuration for the multi-asset price prediction system.

Everything that an examiner may ask "where is this decided?" lives here:
data sources, the frozen chronological split, the prediction target, the
feature policy, default hyper-parameters and which model is served per asset.
"""
import os
import json

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
RAW_LIVE_DIR = os.path.join(BASE_DIR, 'data', 'raw_live')      # live-demo downloads; never touches data/raw
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
TUNING_DIR = os.path.join(RESULTS_DIR, 'tuning')

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, TUNING_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
ASSET_CONFIG = {
    'Bitcoin': {'ticker': 'BTC-USD', 'type': 'crypto',    'source': 'yfinance', 'filename': 'bitcoin_data.csv', 'prefix': 'btc'},
    'Gold':    {'ticker': 'GC=F',    'type': 'commodity', 'source': 'yfinance', 'filename': 'gold_data.csv',    'prefix': 'gold'},
    'Silver':  {'ticker': 'SI=F',    'type': 'commodity', 'source': 'yfinance', 'filename': 'silver_data.csv',  'prefix': 'silver'},
}
ASSETS = list(ASSET_CONFIG.keys())

def get_prefix(asset_name):
    return ASSET_CONFIG[asset_name]['prefix']

# History to download. Yahoo Finance supports much longer histories; more data is
# the single most effective improvement available (see docs/LIMITATIONS.md).
DATA_START_DATE = '2018-01-01'

# ---------------------------------------------------------------------------
# Prediction tasks
# ---------------------------------------------------------------------------
# The served task is the next-day log return  y_t = ln(P_{t+1} / P_t).
# Additional tasks are evaluated as experiments (src/experiments/):
#   return_5d : 5-day forward log return  sum_{k=1..5} r_{t+k}          (overlapping targets → embargo)
#   vol_5d    : log of 5-day forward realised volatility  ln sqrt(mean r²_{t+1..t+5})
TARGET_COL = 'log_return'
PREDICTION_HORIZON_DAYS = 1
TASKS = {
    'return_1d': {'kind': 'return', 'horizon': 1, 'naive_feature': None,       'label': 'Next-day log return'},
    'return_5d': {'kind': 'return', 'horizon': 5, 'naive_feature': None,       'label': '5-day forward log return'},
    'vol_5d':    {'kind': 'vol',    'horizon': 5, 'naive_feature': 'rv_5d',    'label': '5-day forward realised volatility (log)'},
    'vol_22d':   {'kind': 'vol',    'horizon': 22, 'naive_feature': 'rv_22d',  'label': '22-day forward realised volatility (log)'},
}
DEFAULT_TASK = 'return_1d'

# ---------------------------------------------------------------------------
# Frozen chronological split (identical for all assets, by calendar date)
#   train : <= TRAIN_END
#   val   : TRAIN_END < t <= VAL_END        (hyper-parameter / epoch selection, walk-forward folds)
#   test  : > VAL_END                       (touched ONCE, by src/evaluation/backtesting.py)
# ---------------------------------------------------------------------------
TRAIN_END = '2025-09-10'
VAL_END = '2026-02-17'

# ---------------------------------------------------------------------------
# Feature policy
# ---------------------------------------------------------------------------
# Raw / level columns are kept in *_features.csv for charting but are NEVER model inputs
# (non-stationary: their test-set values fall outside the training range).
LEVEL_COLUMNS = ['open', 'high', 'low', 'price', 'volume',
                 'EMA_14', 'EMA_30', 'BB_Mid', 'BB_Upper', 'BB_Lower', 'ATR', 'MACD', 'MACD_Signal']

# Close-time alignment of everything that is not the asset's own close.
#   * BTC-USD: Yahoo's daily bar closes at 00:00 UTC (19:00/20:00 ET), AFTER every US market close,
#     so the same day's S&P 500 / VIX / Fear & Greed values are known when P_t is observed.
#   * GC=F / SI=F: Yahoo's daily close is the 13:25–13:30 ET COMEX settlement (verified against intraday
#     bars), which is BEFORE the S&P (16:00), VIX (16:15), 10-y yield (~15:00), DXY (17:00) and WTI (14:30)
#     closes. The latest macro value known at settlement is therefore the PREVIOUS trading day's.
#     Yahoo's futures High/Low also span the full Globex session (past the settlement), so the features
#     built from them (hl_range, atr_norm, ADX) are lagged one session for the metals.
#   Gold's same-day return stays a same-day feature for Silver: both settle in the same 13:25–13:30 window.
EXTERNAL_SAME_DAY = {'crypto': True, 'commodity': False}
POST_SETTLEMENT_FEATURES = ['hl_range', 'atr_norm', 'ADX']

# Sequence length for the recurrent models (lookback window in trading days)
SEQ_LEN = 30

# ---------------------------------------------------------------------------
# Default hyper-parameters. These are *starting points*; the values actually used
# for the final models come from walk-forward tuning (results/tuning/best_params.json)
# produced by src/training/tune_models.py on train+val only.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    'LightGBM': {'n_estimators': 300, 'learning_rate': 0.02, 'num_leaves': 15,
                 'min_child_samples': 30, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1.0},
    'CatBoost': {'iterations': 500, 'learning_rate': 0.03, 'depth': 4, 'l2_leaf_reg': 3.0},
    'RandomForest': {'n_estimators': 300, 'max_depth': 6, 'min_samples_leaf': 20},
    'Ridge': {'alpha': 10.0},
    'GRU': {'units': 32, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 32, 'epochs': 80, 'patience': 10},
    'LSTM': {'units': 32, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 32, 'epochs': 80, 'patience': 10},
}

BEST_PARAMS_PATH = os.path.join(TUNING_DIR, 'best_params.json')

def get_params(model_name, asset_name=None, task=DEFAULT_TASK):
    """Tuned params for (task, asset, model) if tuning has been run, else defaults.
    best_params.json layout: {task: {asset: {model: params}}} (the served task is 'return_1d')."""
    params = dict(DEFAULT_PARAMS[model_name])
    if asset_name and os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH) as f:
            best = json.load(f)
        params.update(best.get(task, {}).get(asset_name, {}).get(model_name, {}))
    return params

# Walk-forward (expanding window) validation folds inside train+val
CV_FOLDS = 4

# ---------------------------------------------------------------------------
# Which model is served per asset. Written by src/evaluation/backtesting.py based on
# walk-forward validation (NOT test) results; hand-editing is discouraged.
# ---------------------------------------------------------------------------
MODEL_STATUS_PATH = os.path.join(MODELS_DIR, 'model_status.json')
EXPERIMENTS_DIR = os.path.join(RESULTS_DIR, 'experiments')
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

def load_model_status():
    if os.path.exists(MODEL_STATUS_PATH):
        with open(MODEL_STATUS_PATH) as f:
            return json.load(f)
    return {a: {'primary_model': 'LightGBM', 'status': 'untrained'} for a in ASSETS}

MODEL_STATUS = load_model_status()
