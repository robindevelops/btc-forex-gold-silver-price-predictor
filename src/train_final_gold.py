"""
Final Gold LSTM Training Script.

Trains the final Gold price prediction model using the optimal
hyperparameters locked in config.BEST_LSTM_CONFIG (from BTC tuning).

Saves the output to data/models/gold_lstm_final.h5 and natively as .keras.
Also plots the actual vs predicted results.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import MODELS_DIR, PROCESSED_DATA_DIR, BEST_LSTM_CONFIG
from src.model_lstm import build_lstm_model
from src.preprocessing import create_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def inverse_transform_price(scaled_values, scaler, price_col_idx=0, n_features=16):
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, price_col_idx] = np.array(scaled_values).ravel()
    return scaler.inverse_transform(dummy)[:, price_col_idx]


def build_sequences(seq_len):
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'gold_train_scaled.csv'), index_col='timestamp', parse_dates=True)
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'gold_val_scaled.csv'), index_col='timestamp', parse_dates=True)
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'gold_test_scaled.csv'), index_col='timestamp', parse_dates=True)

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
    
    # Extract test dates for plotting (the target date is the row index after the sequence)
    test_dates = full_df.index[L_train + L_val:]
    
    return X_train, y_train, X_val, y_val, X_tr, y_tr, X_test, y_test, test_dates


if __name__ == "__main__":
    cfg = BEST_LSTM_CONFIG
    print(f"\n{'='*50}")
    print("  TRAINING FINAL GOLD MODEL (UNBIASED)")
    print(f"{'='*50}")
    print(f"  Configuration:")
    for k, v in cfg.items():
        print(f"    {k}: {v}")
    
    print("\n  Building continuous sequences...")
    X_train, y_train, X_val, y_val, X_tr, y_tr, X_test, y_test, test_dates = build_sequences(cfg['seq_len'])
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Full Train (Train+Val): {X_tr.shape}")
    
    scaler = joblib.load(os.path.join(MODELS_DIR, 'gold_scaler.pkl'))
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
    
    # --- EVALUATION ON BLIND TEST SET ---
    y_pred_scaled = model_final.predict(X_test, verbose=0)
    rmse_s = np.sqrt(mean_squared_error(y_test, y_pred_scaled))
    mae_s = mean_absolute_error(y_test, y_pred_scaled)
    r2_s = r2_score(y_test, y_pred_scaled)
    
    y_test_real = inverse_transform_price(y_test, scaler, 0, n_features)
    y_pred_real = inverse_transform_price(y_pred_scaled, scaler, 0, n_features)
    rmse_usd = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae_usd = mean_absolute_error(y_test_real, y_pred_real)
    
    print(f"\n  Final UNBIASED Evaluation (Test Set):")
    print(f"    RMSE (scaled): {rmse_s:.6f}")
    print(f"    MAE (scaled):  {mae_s:.6f}")
    print(f"    R²:            {r2_s:.6f}")
    print(f"    RMSE (USD):    ${rmse_usd:,.2f}")
    print(f"    MAE (USD):     ${mae_usd:,.2f}")
    
    # Save the model
    ckpt_path_keras = os.path.join(MODELS_DIR, 'gold_lstm_final.keras')
    ckpt_path_h5 = os.path.join(MODELS_DIR, 'gold_lstm_final.h5')
    
    model_final.save(ckpt_path_keras)
    model_final.save(ckpt_path_h5)
    print(f"\n  ✅ Model saved to {ckpt_path_keras}")
    print(f"  ✅ Model also saved to {ckpt_path_h5}")

    # Plot predictions
    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, y_test_real, label='Actual Gold Price', color='gold', linewidth=2)
    plt.plot(test_dates, y_pred_real, label='LSTM Predicted', color='blue', linestyle='--', linewidth=2)
    plt.title('Gold Price Prediction vs Actual (LSTM Final Test Set)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_path = os.path.join(RESULTS_DIR, 'gold_lstm_predictions.png')
    plt.savefig(plot_path)
    print(f"  ✅ Plot saved to {plot_path}")
    
    # Output markdown format string so Antigravity can read it for artifact
    print(f"\n[EVAL_MARKER]|RMSE={rmse_s:.6f}|MAE={mae_s:.6f}|R2={r2_s:.6f}|RMSE_USD={rmse_usd:.2f}|MAE_USD={mae_usd:.2f}")
