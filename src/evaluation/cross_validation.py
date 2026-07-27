"""
Time-Series Cross Validation Utilities.

Implements Expanding Window CV (Walk-Forward Validation) to replace static train/test splits.
Provides functions to generate out-of-fold predictions for stacking.
"""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def get_cv_splits(n_samples, n_splits=3, test_size=None):
    """
    Generate train and validation indices for expanding window cross-validation.
    
    Args:
        n_samples (int): Total number of sequential samples.
        n_splits (int): Number of CV folds.
        test_size (int, optional): Fixed size for the validation set in each split.
                                   If None, the validation set size will grow or be 
                                   determined by TimeSeriesSplit defaults.
                                   
    Yields:
        tuple: (train_indices, val_indices)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    for train_index, val_index in tscv.split(np.arange(n_samples)):
        yield train_index, val_index

def generate_oof_predictions(model_builder_fn, model_train_fn, model_predict_fn, X, y, n_splits=3):
    """
    Generates out-of-fold (OOF) predictions using Walk-Forward Validation.
    Useful for training a stacked ensemble meta-model on unbiased predictions.
    
    Args:
        model_builder_fn: Function to build/instantiate a fresh model for each fold.
        model_train_fn: Function to train the model, takes (model, X_train, y_train, X_val, y_val).
        model_predict_fn: Function to predict with the trained model, takes (model, X_test).
        X (np.ndarray): Input features (ordered by time).
        y (np.ndarray): Target variable (ordered by time).
        n_splits (int): Number of folds for TimeSeriesSplit.
        
    Returns:
        tuple: (oof_predictions, oof_targets)
            oof_predictions: Array of predictions for the validation periods.
            oof_targets: The true target values corresponding to oof_predictions.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    oof_preds = []
    oof_true = []
    
    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        print(f"    Fold {fold+1}/{n_splits} - Train: {len(train_index)}, Val: {len(val_index)}")
        
        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[val_index], y[val_index]
        
        # Instantiate fresh model
        model = model_builder_fn()
        
        # Train model
        model = model_train_fn(model, X_train, y_train, X_val, y_val)
        
        # Predict on validation set
        preds = model_predict_fn(model, X_val)
        
        # We ensure preds are 1D
        preds = np.array(preds).ravel()
        
        oof_preds.append(preds)
        oof_true.append(y_val.ravel())
        
    # Concatenate all out-of-fold predictions and targets
    # Note: The first chunk of training data is never used as validation in OOF
    return np.concatenate(oof_preds), np.concatenate(oof_true)
