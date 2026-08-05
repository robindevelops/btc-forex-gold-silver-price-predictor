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
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def load_inference_data(asset_name, seq_len=30):
    prefix = 'btc' if asset_name == 'Bitcoin' else asset_name.lower()
    
    test_path = os.path.join(PROCESSED_DATA_DIR, f'{prefix}_test_scaled.csv')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data missing: {test_path}")
        
    test_df = pd.read_csv(test_path, index_col='timestamp', parse_dates=True)
    scaler_path = os.path.join(MODELS_DIR, f'{prefix}_scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler missing: {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    # Target index is either 'log_return' or 'price'
    target_idx = list(test_df.columns).index('log_return') if 'log_return' in test_df.columns else list(test_df.columns).index('price')
    
    # We only need the very last sequence for real-time inference
    # But to test inference, we will take the last 30 days
    last_window = test_df.values[-seq_len:]
    X_latest = np.expand_dims(last_window, axis=0) # shape: (1, seq_len, features)
    
    # Also return DataFrame for PyTorch NeuralForecast if needed
    return X_latest, scaler, target_idx, test_df


def predict_next_day(asset_name):
    logger.info(f"========== PREDICTION INFERENCE: {asset_name.upper()} ==========")
    
    prefix = 'btc' if asset_name == 'Bitcoin' else asset_name.lower()
    seq_len = BEST_LSTM_CONFIG['seq_len']
    X_latest, scaler, target_idx, test_df = load_inference_data(asset_name, seq_len)
    
    status = MODEL_STATUS.get(asset_name, {})
    best_model_type = status.get('primary_model', 'stacked_ensemble')
    
    logger.info(f"Best historical model: {best_model_type}")
    
    try:
        if best_model_type == 'stacked_ensemble':
            # Load Meta-Model
            meta_filename = status.get('model_file', f'{prefix}_meta_model.pkl')
            meta_path = os.path.join(MODELS_DIR, meta_filename)
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Meta-model missing: {meta_path}")
            meta_model = joblib.load(meta_path)
            
            base_preds = []
            base_model_names = status.get('base_models', [])
            
            for base_model_file in base_model_names:
                model_path = os.path.join(MODELS_DIR, base_model_file)
                if not os.path.exists(model_path):
                    logger.warning(f"Base model missing, skipping: {model_path}")
                    continue
                
                if base_model_file.endswith('.pkl'):
                    model = joblib.load(model_path)
                    pred = model.predict(X_latest.reshape(1, -1))[0]
                    base_preds.append(pred)
                elif base_model_file.endswith('.cbm'):
                    from catboost import CatBoostRegressor
                    cb = CatBoostRegressor()
                    cb.load_model(model_path)
                    pred = cb.predict(X_latest[:, -1, :])[0]
                    base_preds.append(pred)
                elif base_model_file.endswith('.keras'):
                    from tensorflow.keras.models import load_model
                    model = load_model(model_path)
                    pred = model.predict(X_latest, verbose=0)[0][0]
                    base_preds.append(pred)
                else:
                    logger.warning(f"Unknown model extension for {base_model_file}")

            if not base_preds:
                raise ValueError("No base models found for the ensemble!")
                
            X_meta = np.array([base_preds])
            final_pred_scaled = meta_model.predict(X_meta)[0]
            
        elif best_model_type == 'gru':
            model_filename = status.get('model_file', f'{prefix}_gru_final.keras')
            model_path = os.path.join(MODELS_DIR, model_filename)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"GRU model missing: {model_path}")
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            final_pred_scaled = model.predict(X_latest, verbose=0)[0][0]
            
        else:
            raise ValueError(f"Inference not implemented for standalone model: {best_model_type}")
            
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise e
        
    # Reconstruct to actual price
    scaled_array = np.array([[final_pred_scaled]])
    real_price = reconstruct_price(scaled_array, X_latest, scaler, target_idx)[0][0]
    
    last_close = test_df['price'].iloc[-1]
    pct_change = ((real_price - last_close) / last_close) * 100
    
    logger.info(f"Last Known Close Price: ${last_close:,.2f}")
    logger.info(f"Predicted Next Day:     ${real_price:,.2f}")
    
    direction = "UP" if pct_change > 0 else "DOWN"
    logger.info(f"Predicted Move:         {direction} ({pct_change:+.2f}%)")
    logger.info("=====================================================")
    
    return {
        'asset': asset_name,
        'current_price': float(last_close),
        'predicted_price': float(real_price),
        'predicted_move_pct': float(pct_change),
        'direction': direction,
        'model_used': best_model_type,
        'metadata': {
            'seq_len': seq_len,
            'ensemble_bases': len(base_preds) if best_model_type == 'stacked_ensemble' else 1
        }
    }


def predict_next_day_safe(asset_name):
    """
    Wrapper for predict_next_day that catches exceptions and returns None 
    instead of crashing the pipeline.
    """
    try:
        return predict_next_day(asset_name)
    except Exception as e:
        logger.error(f"Failed to generate prediction for {asset_name}: {e}")
        return None


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging
    setup_logging()
    predict_next_day_safe('Bitcoin')
    predict_next_day_safe('Gold')
    predict_next_day_safe('Silver')
