"""
Final BTC LSTM Training Script.

Trains the final BTC price prediction model using the optimal
hyperparameters locked in config.BEST_LSTM_CONFIG.

Saves the output to data/models/btc_lstm_final.h5 and natively as .keras.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import MODELS_DIR, PROCESSED_DATA_DIR, BEST_LSTM_CONFIG, BEST_LGBM_CONFIG, BEST_GRU_CONFIG
from src.models.model_lstm import build_lstm_model
from src.models.model_gru import build_gru_model
from src.models.model_lgbm import build_lgbm_model, save_lgbm_model
from src.data.preprocessing import create_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.inverse_transform import reconstruct_price

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


# Replaced by reconstruct_price from src.utils.inverse_transform


def build_sequences(seq_len):
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'btc_train_scaled.csv'), index_col='timestamp', parse_dates=True)
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'btc_val_scaled.csv'), index_col='timestamp', parse_dates=True)
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'btc_test_scaled.csv'), index_col='timestamp', parse_dates=True)

    L_train = len(train_df)
    L_val = len(val_df)
    
    # 1. Create sequences continuously to avoid losing data at the boundaries
    full_df = pd.concat([train_df, val_df, test_df])
    X, y = create_sequences(full_df, seq_len=seq_len)
    
    # 2. Split indices
    split_1 = L_train - seq_len
    split_2 = L_train + L_val - seq_len
    
    X_train, y_train = X[:split_1], y[:split_1]
    X_val, y_val = X[split_1:split_2], y[split_1:split_2]
    X_test, y_test = X[split_2:], y[split_2:]

    # Merge train and val for final retraining
    X_tr = np.concatenate([X_train, X_val], axis=0)
    y_tr = np.concatenate([y_train, y_val], axis=0)
    
    return X_train, y_train, X_val, y_val, X_tr, y_tr, X_test, y_test


if __name__ == "__main__":
    cfg = BEST_LSTM_CONFIG
    print(f"\n{'='*50}")
    print("  TRAINING FINAL BTC MODEL (UNBIASED)")
    print(f"{'='*50}")
    print(f"  Configuration:")
    for k, v in cfg.items():
        print(f"    {k}: {v}")
    
    print("\n  Building continuous sequences...")
    X_train, y_train, X_val, y_val, X_tr, y_tr, X_test, y_test = build_sequences(cfg['seq_len'])
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Full Train (Train+Val): {X_tr.shape}")
    
    scaler = joblib.load(os.path.join(MODELS_DIR, 'btc_scaler.pkl'))
    n_features = X_train.shape[2]
    
    # --- PHASE 1: Find best epoch on validation set (No data leakage) ---
    print("\n  Phase 1: Finding optimal epochs via Validation Set...")
    model_val = build_lstm_model(
        seq_len=cfg['seq_len'], n_features=n_features,
        learning_rate=cfg['learning_rate'], lstm_units=cfg['lstm_units'],
        dense_units=cfg['dense_units'], dropout_rate=cfg['dropout_rate']
    )
    
    callbacks_val = [
        EarlyStopping(monitor='val_loss', patience=cfg['patience'], restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0),
    ]
    
    history_val = model_val.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=cfg['epochs'], batch_size=cfg['batch_size'],
        callbacks=callbacks_val, verbose=0
    )
    
    best_epoch = np.argmin(history_val.history['val_loss']) + 1
    print(f"  Optimal epochs found: {best_epoch} (Early stopping monitored on Val)")
    
    # --- PHASE 2: Retrain blindly on full dataset (Train+Val) for best_epoch ---
    print(f"\n  Phase 2: Retraining on full dataset for {best_epoch} epochs...")
    model_final = build_lstm_model(
        seq_len=cfg['seq_len'], n_features=n_features,
        learning_rate=cfg['learning_rate'], lstm_units=cfg['lstm_units'],
        dense_units=cfg['dense_units'], dropout_rate=cfg['dropout_rate']
    )
    
    # We use a custom learning rate scheduler to mimic the decay that happened during phase 1
    # For simplicity in this script, we'll just train with standard ReduceLROnPlateau but monitor 'loss'
    callbacks_final = [
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0)
    ]
    
    start_time = datetime.now()
    model_final.fit(
        X_tr, y_tr,
        epochs=best_epoch,
        batch_size=cfg['batch_size'],
        callbacks=callbacks_final,
        verbose=0
    )
    print(f"  Training completed in {(datetime.now() - start_time).total_seconds():.0f}s")
    
    # --- EVALUATION ON BLIND TEST SET (LSTM) ---
    y_pred_scaled = model_final.predict(X_test, verbose=0)
    rmse_s = np.sqrt(mean_squared_error(y_test, y_pred_scaled))
    r2_s = r2_score(y_test, y_pred_scaled)
    
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'btc_train_scaled.csv'), index_col=0, nrows=1)
    target_idx = list(train_df.columns).index('log_return') if 'log_return' in train_df.columns else list(train_df.columns).index('price')
    
    y_test_real = reconstruct_price(y_test, X_test, scaler, target_idx)
    y_pred_real = reconstruct_price(y_pred_scaled, X_test, scaler, target_idx)
    rmse_usd = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

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
    # Retrain on full for final? Yes
    model_lgbm.fit(X_tr_flat, y_tr.ravel())
    print(f"  LightGBM Training completed in {(datetime.now() - start_time).total_seconds():.0f}s")
    
    y_pred_lgbm_scaled = model_lgbm.predict(X_test_flat).reshape(-1, 1)
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
    
    gru_path_keras = os.path.join(MODELS_DIR, 'btc_gru_final.keras')
    model_gru_final.save(gru_path_keras)
    print(f"  ✅ GRU saved to {gru_path_keras}")
    
    print("\n" + "="*50)
    print("  MODEL COMPARISON SUMMARY")
    print("="*50)
    print(f"  LSTM:     RMSE ${rmse_usd:,.0f} | R² {r2_s:.4f}")
    print(f"  GRU:      RMSE ${rmse_gru_usd:,.0f} | R² {r2_gru:.4f}")
    print(f"  LightGBM: RMSE ${rmse_lgbm_usd:,.0f} | R² {r2_lgbm:.4f}")
    
    # Save the LSTM model
    ckpt_path_keras = os.path.join(MODELS_DIR, 'btc_lstm_final.keras')
    ckpt_path_h5 = os.path.join(MODELS_DIR, 'btc_lstm_final.h5')
    
    model_final.save(ckpt_path_keras)
    model_final.save(ckpt_path_h5)
    print(f"\n  ✅ LSTM saved to {ckpt_path_keras}")
    print(f"  ✅ LSTM also saved to {ckpt_path_h5}")
