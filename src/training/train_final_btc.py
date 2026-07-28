"""
Final BTC Training Script (Advanced Architecture).

Trains the final BTC price prediction models:
1. PatchTST (PyTorch/NeuralForecast) - State of the art sequence forecasting
2. LightGBM - Advanced tree ensemble
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import MODELS_DIR, PROCESSED_DATA_DIR, BEST_LSTM_CONFIG, BEST_LGBM_CONFIG
from src.models.model_lgbm import build_lgbm_model, save_lgbm_model
from src.data.preprocessing import create_sequences
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.inverse_transform import reconstruct_price

# Import the new advanced models wrapper
from src.models.advanced_models import train_neuralforecast_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def build_sequences(seq_len):
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'btc_train_scaled.csv'), index_col='timestamp', parse_dates=True)
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'btc_val_scaled.csv'), index_col='timestamp', parse_dates=True)
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'btc_test_scaled.csv'), index_col='timestamp', parse_dates=True)

    L_train = len(train_df)
    L_val = len(val_df)
    
    # Create continuous sequences for LightGBM
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
    print("  TRAINING FINAL BTC MODEL (PatchTST + LightGBM)")
    print(f"{'='*50}")
    
    print("\n  Building continuous sequences...")
    X_train, y_train, X_val, y_val, X_tr, y_tr, X_test, y_test, train_df, val_df, test_df = build_sequences(cfg['seq_len'])
    
    scaler = joblib.load(os.path.join(MODELS_DIR, 'btc_scaler.pkl'))
    
    # ---------------------------------------------------------
    # 1. Train PatchTST (Deep Sequence Model)
    # ---------------------------------------------------------
    # We pass the raw dataframes to NeuralForecast to let it build the time-series patches natively
    # We train on Train and validate on Val
    nf_model = train_neuralforecast_model(train_df, val_df, asset_name="Bitcoin", model_type="PatchTST")
    
    # Note: Proper test set evaluation for NeuralForecast requires formatting the test set and passing historical context.
    # We will let the ensemble_model.py handle the formal WFCV evaluation.
    
    # ---------------------------------------------------------
    # 2. Train LightGBM (Tabular Model)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("  TRAINING LIGHTGBM MODEL")
    print("="*50)
    
    lgbm_cfg = BEST_LGBM_CONFIG
    # Flatten sequences for LightGBM
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_tr_flat = X_tr.reshape(X_tr.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    model_lgbm = build_lgbm_model(
        n_estimators=lgbm_cfg['n_estimators'],
        learning_rate=lgbm_cfg['learning_rate'],
        max_depth=lgbm_cfg['max_depth'],
        num_leaves=lgbm_cfg['num_leaves'],
        subsample=lgbm_cfg['subsample'],
        colsample_bytree=lgbm_cfg['colsample_bytree']
    )
    
    start_time = datetime.now()
    # LightGBM handles early stopping natively using eval_set
    model_lgbm.fit(
        X_train_flat, y_train.ravel(),
        eval_set=[(X_val_flat, y_val.ravel())],
        callbacks=[]
    )
    # Retrain on full (Train+Val)
    model_lgbm.fit(X_tr_flat, y_tr.ravel())
    print(f"  LightGBM Training completed in {(datetime.now() - start_time).total_seconds():.0f}s")
    
    y_pred_lgbm_scaled = model_lgbm.predict(X_test_flat).reshape(-1, 1)
    target_idx = list(train_df.columns).index('log_return') if 'log_return' in train_df.columns else list(train_df.columns).index('price')
    y_test_real = reconstruct_price(y_test, X_test, scaler, target_idx)
    y_pred_lgbm_real = reconstruct_price(y_pred_lgbm_scaled, X_test, scaler, target_idx)
    rmse_lgbm_usd = np.sqrt(mean_squared_error(y_test_real, y_pred_lgbm_real))
    r2_lgbm = r2_score(y_test, y_pred_lgbm_scaled)
    
    print(f"\n  Final Evaluation (LightGBM):")
    print(f"    RMSE (USD):    ${rmse_lgbm_usd:,.0f}")
    print(f"    R²:            {r2_lgbm:.6f}")
    
    # Save LightGBM
    lgbm_path = os.path.join(MODELS_DIR, 'btc_lgbm_final.pkl')
    save_lgbm_model(model_lgbm, lgbm_path)
    print(f"  ✅ LightGBM saved to {lgbm_path}")
    
    print("\n  ✅ Bitcoin training complete. Note: Formal ensemble evaluation is handled by ensemble_model.py")
