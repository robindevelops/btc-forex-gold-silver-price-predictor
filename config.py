import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')

# Data sources configuration
ASSET_CONFIG = {
    'Bitcoin': {
        'ticker': 'BTC-USD',
        'type': 'crypto',
        'source': 'yfinance',
        'filename': 'bitcoin_data.csv'
    },
    'Gold': {
        'ticker': 'GC=F',
        'type': 'commodity',
        'source': 'yfinance',
        'filename': 'gold_data.csv'
    },
    'Silver': {
        'ticker': 'SI=F',
        'type': 'commodity',
        'source': 'yfinance',
        'filename': 'silver_data.csv'
    }
}

# Settings mapping
DEFAULT_HISTORY_DAYS = 365 * 3 # 3 years for decent training data

# Best Hyperparameters (found via Week 5/6 experiments)
BEST_LSTM_CONFIG = {
    'seq_len': 30,
    'lstm_units': 100,
    'dense_units': 50,
    'dropout_rate': 0.1,
    'learning_rate': 0.001,
    'batch_size': 16,
    'patience': 20,
    'epochs': 150
}

BEST_LGBM_CONFIG = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 6,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

BEST_GRU_CONFIG = {
    'seq_len': 30,
    'gru_units': 100,
    'dense_units': 50,
    'dropout_rate': 0.1,
    'learning_rate': 0.001,
    'batch_size': 16,
    'patience': 20,
    'epochs': 150
}

# Cross-Validation Configuration (Week 6)
CV_FOLDS = 3  # 3 folds to balance robust evaluation with deep learning training times

# Model status per asset — controls which model is served for predictions.
# Week 5 Update: LightGBM replaces LSTM as primary model.
MODEL_STATUS = {
    'Bitcoin': {
        'primary_model': 'lightgbm',
        'model_file': 'btc_lgbm_final.pkl',
        'status': 'active',
    },
    'Gold': {
        'primary_model': 'lightgbm',
        'model_file': 'gold_lgbm_final.pkl',
        'status': 'active',
    },
    'Silver': {
        'primary_model': 'lightgbm',
        'model_file': 'silver_lgbm_final.pkl',
        'status': 'active',
    },
}

