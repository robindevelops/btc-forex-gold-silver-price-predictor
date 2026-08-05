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

from config import MODELS_DIR, PROCESSED_DATA_DIR, BEST_LSTM_CONFIG, BEST_GRU_CONFIG, BEST_LGBM_CONFIG
from src.models.model_lgbm import build_lgbm_model, save_lgbm_model
from src.models.model_gru import build_gru_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.data.preprocessing import create_sequences
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.inverse_transform import reconstruct_price

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
    print("  TRAINING FINAL GOLD MODEL (GRU + CatBoost + LightGBM)")
    print(f"{'='*50}")
    
    print("\n  Building continuous sequences...")
    X_train, y_train, X_val, y_val, X_tr, y_tr, X_test, y_test, train_df, val_df, test_df = build_sequences(cfg['seq_len'])
    
    scaler = joblib.load(os.path.join(MODELS_DIR, 'gold_scaler.pkl'))
    
    # ---------------------------------------------------------
    # 1. Train GRU (Deep Sequence Model)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("  TRAINING GRU MODEL")
    print("="*50)

    gru_cfg = {k: v for k, v in BEST_GRU_CONFIG.items() if k not in ['seq_len', 'batch_size', 'epochs', 'patience']}
    model_gru = build_gru_model(seq_len=cfg['seq_len'], n_features=X_train.shape[2], **gru_cfg)

    early_stop = EarlyStopping(monitor='val_loss', patience=BEST_GRU_CONFIG.get('patience', 10), restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    history = model_gru.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=BEST_GRU_CONFIG.get('epochs', 50),
        batch_size=BEST_GRU_CONFIG.get('batch_size', 32),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    optimal_epochs = max(1, len(history.history['loss']) - BEST_GRU_CONFIG.get('patience', 10))
    print(f"  Optimal epochs found: {optimal_epochs}")
    
    model_gru = build_gru_model(seq_len=cfg['seq_len'], n_features=X_tr.shape[2], **gru_cfg)
    model_gru.fit(
        X_tr, y_tr,
        epochs=optimal_epochs,
        batch_size=BEST_GRU_CONFIG.get('batch_size', 32),
        verbose=1
    )
    
    gru_path = os.path.join(MODELS_DIR, 'gold_gru_final.keras')
    model_gru.save(gru_path)
    print(f"  ✅ GRU saved to {gru_path}")
    
    # ---------------------------------------------------------
    # 2. Train CatBoost (Advanced Tabular Model)
    # ---------------------------------------------------------
    model_catboost = train_catboost(X_tr, y_tr, X_val, y_val, asset_name="Gold")
    
    # ---------------------------------------------------------
    # 3. Train LightGBM (Tabular Model)
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
    
    lgbm_path = os.path.join(MODELS_DIR, 'gold_lgbm_final.pkl')
    save_lgbm_model(model_lgbm, lgbm_path)
    print(f"  ✅ LightGBM saved to {lgbm_path}")
    
    print("\n  ✅ Gold training complete. Note: Formal ensemble evaluation is handled by ensemble_model.py")
