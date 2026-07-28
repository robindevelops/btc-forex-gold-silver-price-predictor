"""
Final Gold Training Script (Advanced Architecture).

Trains the final Gold price prediction models:
1. TFT (Temporal Fusion Transformer) - State of the art macro-covariate model
2. CatBoost - Advanced tabular model
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import MODELS_DIR, PROCESSED_DATA_DIR, BEST_LSTM_CONFIG
from src.data.preprocessing import create_sequences
from src.utils.inverse_transform import reconstruct_price

# Import the new advanced models wrapper
from src.models.advanced_models import train_neuralforecast_model
from src.models.catboost_model import train_catboost

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_sequences(seq_len):
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'gold_train_scaled.csv'), index_col='timestamp', parse_dates=True)
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'gold_val_scaled.csv'), index_col='timestamp', parse_dates=True)
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'gold_test_scaled.csv'), index_col='timestamp', parse_dates=True)

    L_train = len(train_df)
    L_val = len(val_df)
    
    full_df = pd.concat([train_df, val_df, test_df])
    X, y = create_sequences(full_df, seq_len=seq_len)
    
    split_1 = L_train - seq_len
    split_2 = L_train + L_val - seq_len
    
    X_train, y_train = X[:split_1], y[:split_1]
    X_val, y_val = X[split_1:split_2], y[split_1:split_2]
    X_test, y_test = X[split_2:], y[split_2:]

    X_tr = np.concatenate([X_train, X_val], axis=0)
    y_tr = np.concatenate([y_train, y_val], axis=0)
    
    return X_train, y_train, X_val, y_val, X_tr, y_tr, X_test, y_test, train_df, val_df, test_df


if __name__ == "__main__":
    cfg = BEST_LSTM_CONFIG
    print(f"\n{'='*50}")
    print("  TRAINING FINAL GOLD MODEL (TFT + CatBoost)")
    print(f"{'='*50}")
    
    print("\n  Building continuous sequences...")
    X_train, y_train, X_val, y_val, X_tr, y_tr, X_test, y_test, train_df, val_df, test_df = build_sequences(cfg['seq_len'])
    
    scaler = joblib.load(os.path.join(MODELS_DIR, 'gold_scaler.pkl'))
    
    # ---------------------------------------------------------
    # 1. Train TFT (Deep Macro-covariate Model)
    # ---------------------------------------------------------
    nf_model = train_neuralforecast_model(train_df, val_df, asset_name="Gold", model_type="TFT")
    
    # ---------------------------------------------------------
    # 2. Train CatBoost (Advanced Tabular Model)
    # ---------------------------------------------------------
    model_catboost = train_catboost(X_tr, y_tr, X_val, y_val, asset_name="Gold")
    
    print("\n  ✅ Gold training complete. Note: Formal ensemble evaluation is handled by ensemble_model.py")
