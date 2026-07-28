import os
import joblib
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_catboost(X_train, y_train, X_val, y_val, asset_name, feature_names=None):
    """
    Trains a CatBoost Regressor. 
    CatBoost is heavily optimized for tabular data and handles non-linearities exceptionally well.
    Since we use a 30-day sequence for RNNs, for CatBoost we flatten the last day's features 
    (or take the mean of the sequence) to make it 2D.
    """
    print(f"\n{'='*50}\n  TRAINING CATBOOST MODEL ({asset_name})\n{'='*50}")
    
    # Flatten 3D (samples, seq_len, features) to 2D (samples, features)
    # We take the LAST day of the sequence as it holds the most recent technical indicators
    if len(X_train.shape) == 3:
        X_train_2d = X_train[:, -1, :]
        X_val_2d = X_val[:, -1, :]
    else:
        X_train_2d = X_train
        X_val_2d = X_val
        
    y_train_1d = y_train.ravel()
    y_val_1d = y_val.ravel()
    
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        eval_metric='RMSE',
        verbose=100,
        early_stopping_rounds=50,
        random_seed=42
    )
    
    model.fit(
        X_train_2d, y_train_1d,
        eval_set=(X_val_2d, y_val_1d),
        use_best_model=True
    )
    
    # Predict and evaluate
    val_preds = model.predict(X_val_2d)
    rmse = np.sqrt(mean_squared_error(y_val_1d, val_preds))
    r2 = r2_score(y_val_1d, val_preds)
    
    print(f"\n  Final Evaluation (CatBoost):")
    print(f"    RMSE:    {rmse:.4f}")
    print(f"    R²:      {r2:.4f}")
    
    # Save model
    from config import MODELS_DIR
    model_path = os.path.join(MODELS_DIR, f"{asset_name.lower()}_catboost_final.cbm")
    model.save_model(model_path)
    print(f"  ✅ CatBoost saved to {model_path}")
    
    return model
