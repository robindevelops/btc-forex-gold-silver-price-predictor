"""
LightGBM Model Architecture.

Week 5: Tabular Tree-based Model.
LightGBM is highly robust on tabular data with small sample sizes (~700 days).
It handles non-linearities and feature interactions better than Linear Regression
and avoids the massive overfitting risks of deep neural networks (LSTM).
"""

import lightgbm as lgb
import os
import joblib

def build_lgbm_model(n_estimators=100, learning_rate=0.05, max_depth=6,
                     num_leaves=31, subsample=0.8, colsample_bytree=0.8):
    """
    Builds a LightGBM Regressor configured for time-series / tabular data.
    
    Args:
        n_estimators (int): Number of boosting rounds.
        learning_rate (float): Step size for each boosting round.
        max_depth (int): Max tree depth (controls overfitting).
        num_leaves (int): Max leaves per tree (controls complexity).
        subsample (float): Fraction of samples to use per tree.
        colsample_bytree (float): Fraction of features to use per tree.
        
    Returns:
        lgb.LGBMRegressor: Configured model instance.
    """
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        n_jobs=-1,
        verbose=-1 # Suppress lightgbm warnings/info
    )
    return model

def save_lgbm_model(model, filepath):
    """Saves the LightGBM model using joblib."""
    joblib.dump(model, filepath)

def load_lgbm_model(filepath):
    """Loads a saved LightGBM model."""
    return joblib.load(filepath)
