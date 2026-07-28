"""
Advanced Multi-Asset Prediction Pipeline (PyTorch / Keras / LightGBM / CatBoost).

Dynamically loads the best Meta-Model (Stacked Ensemble) or Base Model 
for each asset and generates predictions using Walk-Forward Scalers.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import MODELS_DIR, PROCESSED_DATA_DIR, MODEL_STATUS, BEST_LSTM_CONFIG, BEST_LGBM_CONFIG
from src.data.preprocessing import create_sequences
from src.utils.inverse_transform import reconstruct_price
from tensorflow.keras.models import load_model

def load_inference_data(asset_name, seq_len=30):
    prefix = 'btc' if asset_name == 'Bitcoin' else asset_name.lower()
    
    test_path = os.path.join(PROCESSED_DATA_DIR, f'{prefix}_test_scaled.csv')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data missing: {test_path}")
        
    test_df = pd.read_csv(test_path, index_col='timestamp', parse_dates=True)
    scaler = joblib.load(os.path.join(MODELS_DIR, f'{prefix}_scaler.pkl'))
    
    # Target index is either 'log_return' or 'price'
    target_idx = list(test_df.columns).index('log_return') if 'log_return' in test_df.columns else list(test_df.columns).index('price')
    
    # We only need the very last sequence for real-time inference
    # But to test inference, we will take the last 30 days
    last_window = test_df.values[-seq_len:]
    X_latest = np.expand_dims(last_window, axis=0) # shape: (1, seq_len, features)
    
    # Also return DataFrame for PyTorch NeuralForecast if needed
    return X_latest, scaler, target_idx, test_df


def predict_next_day(asset_name):
    print(f"\n{'='*50}")
    print(f"  PREDICTION INFERENCE: {asset_name.upper()}")
    print(f"{'='*50}")
    
    prefix = 'btc' if asset_name == 'Bitcoin' else asset_name.lower()
    seq_len = BEST_LSTM_CONFIG['seq_len']
    X_latest, scaler, target_idx, test_df = load_inference_data(asset_name, seq_len)
    
    status = MODEL_STATUS.get(asset_name, {})
    best_model_type = status.get('best_model', 'Stacked Ensemble')
    
    print(f"  Best historical model: {best_model_type}")
    
    try:
        if best_model_type == 'Stacked Ensemble':
            # Load Meta-Model
            meta_path = os.path.join(MODELS_DIR, f'{prefix}_metamodel.pkl')
            meta_model = joblib.load(meta_path)
            
            base_preds = []
            
            # Predict with LightGBM
            lgbm_path = os.path.join(MODELS_DIR, f'{prefix}_lgbm_final.pkl')
            if os.path.exists(lgbm_path):
                lgbm = joblib.load(lgbm_path)
                lgbm_pred = lgbm.predict(X_latest.reshape(1, -1))[0]
                base_preds.append(lgbm_pred)
            
            # Predict with CatBoost
            cat_path = os.path.join(MODELS_DIR, f'{prefix}_catboost_final.cbm')
            if os.path.exists(cat_path):
                from catboost import CatBoostRegressor
                cb = CatBoostRegressor()
                cb.load_model(cat_path)
                cb_pred = cb.predict(X_latest[:, -1, :])[0]
                base_preds.append(cb_pred)
                
            # Predict with GRU (Fallback for Deep Learning)
            gru_path = os.path.join(MODELS_DIR, f'{prefix}_gru_final.keras')
            if os.path.exists(gru_path):
                gru = load_model(gru_path)
                gru_pred = gru.predict(X_latest, verbose=0)[0][0]
                base_preds.append(gru_pred)
                
            # NeuralForecast models are complex to infer point-in-time dynamically without the full NeuralForecast object
            # So if we have PatchTST or TFT, we would load it here.
            # For this pipeline, we will use the base_preds we successfully loaded.
            
            if not base_preds:
                raise ValueError("No base models found for the ensemble!")
                
            X_meta = np.array([base_preds])
            final_pred_scaled = meta_model.predict(X_meta)[0]
            
        elif best_model_type == 'GRU':
            model_path = os.path.join(MODELS_DIR, f'{prefix}_gru_final.keras')
            model = load_model(model_path)
            final_pred_scaled = model.predict(X_latest, verbose=0)[0][0]
            
        else:
            raise ValueError(f"Inference not implemented for standalone model: {best_model_type}")
            
    except Exception as e:
        print(f"  ❌ Error during inference: {e}")
        return
        
    # Reconstruct to actual price
    scaled_array = np.array([[final_pred_scaled]])
    real_price = reconstruct_price(scaled_array, X_latest, scaler, target_idx)[0][0]
    
    last_close = test_df['price'].iloc[-1]
    pct_change = ((real_price - last_close) / last_close) * 100
    
    print(f"\n  Last Known Close Price: ${last_close:,.2f}")
    print(f"  Predicted Next Day:     ${real_price:,.2f}")
    
    direction = "📈 UP" if pct_change > 0 else "📉 DOWN"
    print(f"  Predicted Move:         {direction} ({pct_change:+.2f}%)")
    print(f"{'='*50}\n")
    
    return real_price


if __name__ == "__main__":
    predict_next_day('Bitcoin')
    predict_next_day('Gold')
    predict_next_day('Silver')
