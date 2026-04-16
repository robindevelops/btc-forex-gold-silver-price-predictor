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
