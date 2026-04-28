"""
Unified Evaluation Module.

Computes RMSE, MAE, R², and MAPE (Mean Absolute Percentage Error) cleanly
for any model (LSTM or Baseline). 
Also performs the final side-by-side comparison of 3 LSTMs vs 6 Baselines (9 models total).
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import PROCESSED_DATA_DIR, MODELS_DIR, BEST_LSTM_CONFIG
from src.data.preprocessing import create_sequences

def compute_mape(y_true, y_pred):
    # Avoid division by zero
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def inverse_transform_price(scaled_values, scaler, price_col_idx=0, n_features=16):
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, price_col_idx] = np.array(scaled_values).ravel()
    return scaler.inverse_transform(dummy)[:, price_col_idx]

def get_asset_prefix(asset_name):
    return 'btc' if asset_name == 'Bitcoin' else asset_name.lower()

def evaluate_predictions(y_test, y_pred, y_test_real, y_pred_real):
    rmse_s = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_s = mean_absolute_error(y_test, y_pred)
    r2_s = r2_score(y_test, y_pred)
    mape_s = compute_mape(y_test, y_pred)
    
    rmse_u = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae_u = mean_absolute_error(y_test_real, y_pred_real)
    mape_u = compute_mape(y_test_real, y_pred_real)
    
    return {
        'RMSE': rmse_s,
        'MAE': mae_s,
        'R2': r2_s,
        'MAPE': mape_s,
        'RMSE_USD': rmse_u,
        'MAE_USD': mae_u,
        'MAPE_USD': mape_u
    }

def evaluate_lstm(asset_name):
    prefix = get_asset_prefix(asset_name)
    model_path = os.path.join(MODELS_DIR, f"{prefix}_lstm_final.keras")
    scaler_path = os.path.join(MODELS_DIR, f"{prefix}_scaler.pkl")
    
    if not os.path.exists(model_path):
        return None
        
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    seq_len = BEST_LSTM_CONFIG['seq_len']
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_train_scaled.csv'), index_col='timestamp', parse_dates=True)
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_val_scaled.csv'), index_col='timestamp', parse_dates=True)
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_test_scaled.csv'), index_col='timestamp', parse_dates=True)
    
    full_df = pd.concat([train_df, val_df, test_df])
    X, y = create_sequences(full_df, seq_len=seq_len)
    
    split_2 = len(train_df) + len(val_df) - seq_len
    X_test, y_test = X[split_2:], y[split_2:]
    
    y_pred = model.predict(X_test, verbose=0)
    n_features = X_test.shape[2]
    
    y_test_real = inverse_transform_price(y_test, scaler, 0, n_features)
    y_pred_real = inverse_transform_price(y_pred, scaler, 0, n_features)
    
    res = evaluate_predictions(y_test, y_pred, y_test_real, y_pred_real)
    res['Asset'] = asset_name
    res['Model'] = 'LSTM (Optimized)'
    return res

def evaluate_baseline(asset_name, model_type):
    prefix = get_asset_prefix(asset_name)
    scaler = joblib.load(os.path.join(MODELS_DIR, f"{prefix}_scaler.pkl"))
    
    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_test.npy"))
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    if model_type == 'LinearRegression':
        model = LinearRegression()
    elif model_type == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
    model.fit(X_train_flat, y_train.ravel())
    y_pred = model.predict(X_test_flat).reshape(-1, 1)
    
    n_features = X_train.shape[2]
    y_test_real = inverse_transform_price(y_test, scaler, 0, n_features)
    y_pred_real = inverse_transform_price(y_pred, scaler, 0, n_features)
    
    res = evaluate_predictions(y_test, y_pred, y_test_real, y_pred_real)
    res['Asset'] = asset_name
    res['Model'] = f"Baseline ({model_type})"
    return res

if __name__ == "__main__":
    assets = ['Bitcoin', 'Gold', 'Silver']
    baselines = ['LinearRegression', 'RandomForest']
    
    results = []
    print("Evaluating models... this will take a moment.")
    for asset in assets:
        print(f"  -> {asset}")
        res_lstm = evaluate_lstm(asset)
        if res_lstm:
            results.append(res_lstm)
            
        for b in baselines:
            res_b = evaluate_baseline(asset, b)
            results.append(res_b)
            
    df = pd.DataFrame(results)
    
    # Reorder columns for readability
    cols = ['Asset', 'Model', 'RMSE_USD', 'MAE_USD', 'MAPE_USD', 'R2', 'RMSE']
    df = df[cols]
    
    print("\n" + "="*90)
    print("FINAL 9-MODEL COMPARISON TABLE (3 LSTMs vs 6 Baselines)")
    print("="*90)
    
    for asset in assets:
        asset_df = df[df['Asset'] == asset].sort_values('RMSE_USD')
        print(f"\n--- {asset.upper()} ---")
        # Format the numbers nicely for the print output
        print_df = asset_df.copy()
        print_df['RMSE_USD'] = print_df['RMSE_USD'].apply(lambda x: f"${x:,.2f}")
        print_df['MAE_USD'] = print_df['MAE_USD'].apply(lambda x: f"${x:,.2f}")
        print_df['MAPE_USD'] = print_df['MAPE_USD'].apply(lambda x: f"{x:.2f}%")
        print_df['R2'] = print_df['R2'].apply(lambda x: f"{x:.4f}")
        print_df['RMSE'] = print_df['RMSE'].apply(lambda x: f"{x:.4f}")
        print(print_df.to_string(index=False))
        
    out_path = os.path.join(os.path.dirname(MODELS_DIR), '..', 'results', 'final_9model_comparison.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved full comparison to {out_path}")
