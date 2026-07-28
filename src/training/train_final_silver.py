"""
Final Silver Training Script (Advanced Architecture).

Trains the final Silver price prediction models:
1. LightGBM - Advanced tree ensemble for noise robustness
2. CatBoost - Advanced tabular model
3. GRU - Recurrent model
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import MODELS_DIR, PROCESSED_DATA_DIR, BEST_GRU_CONFIG, BEST_LGBM_CONFIG
from src.models.model_gru import build_gru_model
from src.models.model_lgbm import build_lgbm_model, save_lgbm_model
from src.data.preprocessing import create_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score
from src.utils.inverse_transform import reconstruct_price

# Import the new advanced models wrapper
from src.models.catboost_model import train_catboost

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_sequences(seq_len):
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'silver_train_scaled.csv'), index_col='timestamp', parse_dates=True)
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'silver_val_scaled.csv'), index_col='timestamp', parse_dates=True)
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'silver_test_scaled.csv'), index_col='timestamp', parse_dates=True)

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
    cfg = BEST_GRU_CONFIG
    print(f"\n{'='*50}")
    print("  TRAINING FINAL SILVER MODEL (LightGBM + CatBoost + GRU)")
    print(f"{'='*50}")
    
    print("\n  Building continuous sequences...")
    X_train, y_train, X_val, y_val, X_tr, y_tr, X_test, y_test, train_df, val_df, test_df = build_sequences(cfg['seq_len'])
    
    scaler = joblib.load(os.path.join(MODELS_DIR, 'silver_scaler.pkl'))
    n_features = X_train.shape[2]
    
    # ---------------------------------------------------------
    # 1. Train LightGBM
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("  TRAINING LIGHTGBM MODEL")
    print("="*50)
    
    lgbm_cfg = BEST_LGBM_CONFIG
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
    model_lgbm.fit(
        X_train_flat, y_train.ravel(),
        eval_set=[(X_val_flat, y_val.ravel())],
        callbacks=[]
    )
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
    
    lgbm_path = os.path.join(MODELS_DIR, 'silver_lgbm_final.pkl')
    save_lgbm_model(model_lgbm, lgbm_path)
    print(f"  ✅ LightGBM saved to {lgbm_path}")
    
    # ---------------------------------------------------------
    # 2. Train CatBoost
    # ---------------------------------------------------------
    model_catboost = train_catboost(X_tr, y_tr, X_val, y_val, asset_name="Silver")

    # ---------------------------------------------------------
    # 3. Train GRU
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("  TRAINING GRU MODEL")
    print("="*50)
    
    gru_cfg = BEST_GRU_CONFIG
    
    print("\n  Phase 1: Finding optimal epochs via Validation Set...")
    model_gru_val = build_gru_model(
        seq_len=gru_cfg['seq_len'], n_features=n_features,
        learning_rate=gru_cfg['learning_rate'], gru_units=gru_cfg['gru_units'],
        dense_units=gru_cfg['dense_units'], dropout_rate=gru_cfg['dropout_rate']
    )
    callbacks_val = [
        EarlyStopping(monitor='val_loss', patience=gru_cfg['patience'], restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0),
    ]
    history_gru_val = model_gru_val.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=gru_cfg['epochs'], batch_size=gru_cfg['batch_size'],
        callbacks=callbacks_val, verbose=0
    )
    best_epoch_gru = np.argmin(history_gru_val.history['val_loss']) + 1
    print(f"  Optimal GRU epochs found: {best_epoch_gru}")
    
    print(f"\n  Phase 2: Retraining on full dataset for {best_epoch_gru} epochs...")
    model_gru_final = build_gru_model(
        seq_len=gru_cfg['seq_len'], n_features=n_features,
        learning_rate=gru_cfg['learning_rate'], gru_units=gru_cfg['gru_units'],
        dense_units=gru_cfg['dense_units'], dropout_rate=gru_cfg['dropout_rate']
    )
    callbacks_final = [
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0)
    ]
    start_time = datetime.now()
    model_gru_final.fit(
        X_tr, y_tr,
        epochs=best_epoch_gru,
        batch_size=gru_cfg['batch_size'],
        callbacks=callbacks_final,
        verbose=0
    )
    print(f"  GRU Training completed in {(datetime.now() - start_time).total_seconds():.0f}s")
    
    y_pred_gru_scaled = model_gru_final.predict(X_test, verbose=0)
    y_pred_gru_real = reconstruct_price(y_pred_gru_scaled, X_test, scaler, target_idx)
    rmse_gru_usd = np.sqrt(mean_squared_error(y_test_real, y_pred_gru_real))
    r2_gru = r2_score(y_test, y_pred_gru_scaled)
    
    print(f"\n  Final Evaluation (GRU):")
    print(f"    RMSE (USD):    ${rmse_gru_usd:,.0f}")
    print(f"    R²:            {r2_gru:.6f}")
    
    gru_path_keras = os.path.join(MODELS_DIR, 'silver_gru_final.keras')
    model_gru_final.save(gru_path_keras)
    print(f"  ✅ GRU saved to {gru_path_keras}")
    
    print("\n  ✅ Silver training complete. Note: Formal ensemble evaluation is handled by ensemble_model.py")
