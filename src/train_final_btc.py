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
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'btc_train_scaled.csv'), index_col='timestamp', parse_dates=True)
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'btc_val_scaled.csv'), index_col='timestamp', parse_dates=True)
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'btc_test_scaled.csv'), index_col='timestamp', parse_dates=True)

    X_train, y_train = create_sequences(train_df, seq_len=seq_len)
    X_val, y_val = create_sequences(val_df, seq_len=seq_len)
    X_test, y_test = create_sequences(test_df, seq_len=seq_len)

    # Merge train and val
    X_tr = np.concatenate([X_train, X_val], axis=0)
    y_tr = np.concatenate([y_train, y_val], axis=0)
    return X_tr, y_tr, X_test, y_test


if __name__ == "__main__":
    cfg = BEST_LSTM_CONFIG
    print(f"\n{'='*50}")
    print("  TRAINING FINAL BTC MODEL")
    print(f"{'='*50}")
    print(f"  Configuration:")
    for k, v in cfg.items():
        print(f"    {k}: {v}")
    
    print("\n  Building sequences...")
    X_train, y_train, X_test, y_test = build_sequences(cfg['seq_len'])
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    scaler = joblib.load(os.path.join(MODELS_DIR, 'btc_scaler.pkl'))
    n_features = X_train.shape[2]
    
    print("\n  Building model...")
    model = build_lstm_model(
        seq_len=cfg['seq_len'],
        n_features=n_features,
        learning_rate=cfg['learning_rate'],
        lstm_units=cfg['lstm_units'],
        dense_units=cfg['dense_units'],
        dropout_rate=cfg['dropout_rate']
    )
    
    # Save the model natively to .keras but also as .h5
    ckpt_path_keras = os.path.join(MODELS_DIR, 'btc_lstm_final.keras')
    ckpt_path_h5 = os.path.join(MODELS_DIR, 'btc_lstm_final.h5')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=cfg['patience'], restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=ckpt_path_keras, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]
    
    start_time = datetime.now()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=cfg['epochs'],
        batch_size=cfg['batch_size'],
        callbacks=callbacks,
        verbose=0
    )
    
    print(f"\n  Training completed in {(datetime.now() - start_time).total_seconds():.0f}s")
    
    # Evaluate
    y_pred_scaled = model.predict(X_test, verbose=0)
    rmse_s = np.sqrt(mean_squared_error(y_test, y_pred_scaled))
    r2_s = r2_score(y_test, y_pred_scaled)
    
    y_test_real = inverse_transform_price(y_test, scaler, 0, n_features)
    y_pred_real = inverse_transform_price(y_pred_scaled, scaler, 0, n_features)
    rmse_usd = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    
    print(f"  Final Evaluation:")
    print(f"    RMSE (scaled): {rmse_s:.6f}")
    print(f"    R²:            {r2_s:.6f}")
    print(f"    RMSE (USD):    ${rmse_usd:,.0f}")
    
    # Also save as h5 per user request
    model.save(ckpt_path_h5)
    print(f"\n  ✅ Model saved to {ckpt_path_keras}")
    print(f"  ✅ Model also saved to {ckpt_path_h5}")
