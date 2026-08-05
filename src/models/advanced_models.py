"""
Advanced Models Factory Module.
Provides factory functions for creating, training, and predicting with 
GRU, LightGBM, and CatBoost models.
"""

import logging
from typing import Any, Tuple, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.models.model_gru import build_gru_model
from src.models.model_lgbm import build_lgbm_model
from src.models.catboost_model import train_catboost
from config import BEST_GRU_CONFIG, BEST_LGBM_CONFIG


def get_model(model_type: str, **kwargs: Any) -> Any:
    """
    Factory function to get an instance of the specified model.
    
    Args:
        model_type: "GRU", "LightGBM", or "CatBoost"
        **kwargs: Additional parameters to override defaults
        
    Returns:
        The instantiated model.
    """
    if model_type == "GRU":
        seq_len = kwargs.get('seq_len', 30)
        n_features = kwargs.get('n_features', 1)
        config = {k: v for k, v in BEST_GRU_CONFIG.items() if k not in ['seq_len', 'batch_size', 'epochs', 'patience']}
        for k, v in kwargs.items():
            if k in config:
                config[k] = v
        return build_gru_model(seq_len=seq_len, n_features=n_features, **config)
    elif model_type == "LightGBM":
        config = BEST_LGBM_CONFIG.copy()
        for k, v in kwargs.items():
            if k in config:
                config[k] = v
        return build_lgbm_model(**config)
    elif model_type == "CatBoost":
        from catboost import CatBoostRegressor
        # Using a default configured CatBoost model
        iterations = kwargs.get('iterations', 100)
        learning_rate = kwargs.get('learning_rate', 0.05)
        depth = kwargs.get('depth', 6)
        return CatBoostRegressor(iterations=iterations, learning_rate=learning_rate, depth=depth, verbose=0)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_model(model: Any, X_train: np.ndarray, y_train: np.ndarray, 
                X_val: Optional[np.ndarray], y_val: Optional[np.ndarray], 
                model_type: str, **kwargs: Any) -> Any:
    """
    Trains the specified model.
    
    Args:
        model: The model to train.
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features (optional).
        y_val: Validation targets (optional).
        model_type: Type of the model ("GRU", "LightGBM", "CatBoost").
        **kwargs: Additional training parameters (e.g. epochs, batch_size, callbacks).
        
    Returns:
        The trained model.
    """
    logger.info(f"Training {model_type} model...")
    if model_type == "GRU":
        epochs = kwargs.get('epochs', BEST_GRU_CONFIG.get('epochs', 20))
        batch_size = kwargs.get('batch_size', BEST_GRU_CONFIG.get('batch_size', 16))
        callbacks = kwargs.get('callbacks', [])
        
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                      epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
        else:
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
        return model
    elif model_type == "LightGBM":
        # Ensure 2D input for tabular models
        if len(X_train.shape) == 3:
            X_tr_flat = X_train.reshape(X_train.shape[0], -1)
            X_vl_flat = X_val.reshape(X_val.shape[0], -1) if X_val is not None else None
        else:
            X_tr_flat, X_vl_flat = X_train, X_val
            
        if X_vl_flat is not None and y_val is not None:
            model.fit(X_tr_flat, y_train.ravel(), eval_set=[(X_vl_flat, y_val.ravel())])
        else:
            model.fit(X_tr_flat, y_train.ravel())
        return model
    elif model_type == "CatBoost":
        if len(X_train.shape) == 3:
            X_tr_flat = X_train[:, -1, :]  # Taking the last step like in the codebase
            X_vl_flat = X_val[:, -1, :] if X_val is not None else None
        else:
            X_tr_flat, X_vl_flat = X_train, X_val
            
        if X_vl_flat is not None and y_val is not None:
            model.fit(X_tr_flat, y_train.ravel(), eval_set=(X_vl_flat, y_val.ravel()))
        else:
            model.fit(X_tr_flat, y_train.ravel())
        return model
    else:
        raise ValueError(f"Unknown model_type for training: {model_type}")


def predict_model(model: Any, X: np.ndarray, model_type: str) -> np.ndarray:
    """
    Generates predictions using the specified model.
    
    Args:
        model: The trained model.
        X: Features to predict on.
        model_type: Type of the model ("GRU", "LightGBM", "CatBoost").
        
    Returns:
        Numpy array of predictions.
    """
    if model_type == "GRU":
        return model.predict(X, verbose=0).reshape(-1, 1)
    elif model_type == "LightGBM":
        if len(X.shape) == 3:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
        return model.predict(X_flat).reshape(-1, 1)
    elif model_type == "CatBoost":
        if len(X.shape) == 3:
            X_flat = X[:, -1, :]
        else:
            X_flat = X
        return model.predict(X_flat).reshape(-1, 1)
    else:
        raise ValueError(f"Unknown model_type for prediction: {model_type}")
