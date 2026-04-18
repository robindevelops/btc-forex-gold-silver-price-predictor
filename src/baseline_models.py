"""
Baseline Models for BTC & Forex Price Prediction.

Linear Regression serves as the simplest baseline. If the LSTM can't
beat this, something is wrong with the architecture or data pipeline.
"""

import numpy as np
import pandas as pd
import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except (ImportError, OSError):
    HAS_XGBOOST = False
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PROCESSED_DATA_DIR

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def flatten_sequences(X):
    """
    Flatten 3D LSTM sequences (samples, timesteps, features)
    into 2D sklearn input (samples, timesteps * features).
    """
    return X.reshape(X.shape[0], -1)


def run_linear_regression(asset_name='Bitcoin'):
    """
    Trains a Linear Regression baseline on the given asset and
    evaluates RMSE, MAE, R² on the test set.
    """
    # Determine file prefix
    prefix = 'btc' if asset_name == 'Bitcoin' else asset_name.lower()

    # Load saved .npy sequences
    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_train.npy"))
    X_val   = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_val.npy"))
    y_val   = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_val.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_test.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_test.npy"))

    print(f"\n{'='*50}")
    print(f"Linear Regression Baseline — {asset_name}")
    print(f"{'='*50}")
    print(f"Original shapes: X_train {X_train.shape}, X_test {X_test.shape}")

    # Flatten 3D → 2D for sklearn
    X_train_flat = flatten_sequences(X_train)
    X_val_flat   = flatten_sequences(X_val)
    X_test_flat  = flatten_sequences(X_test)

    print(f"Flattened shapes: X_train {X_train_flat.shape}, X_test {X_test_flat.shape}")

    # Train
    model = LinearRegression()
    model.fit(X_train_flat, y_train.ravel())
    print("Model trained.")

    # Predict on test set
    y_pred = model.predict(X_test_flat)

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\n--- Test Set Metrics ---")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.6f}")

    return {
        'asset': asset_name,
        'model': 'LinearRegression',
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'train_samples': X_train_flat.shape[0],
        'test_samples': X_test_flat.shape[0],
    }


def run_random_forest(asset_name='Bitcoin', n_estimators=100):
    """
    Trains a Random Forest Regressor baseline on the given asset and
    evaluates RMSE, MAE, R² on the test set.
    """
    prefix = 'btc' if asset_name == 'Bitcoin' else asset_name.lower()

    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_test.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_test.npy"))

    print(f"\n{'='*50}")
    print(f"Random Forest Baseline — {asset_name}")
    print(f"{'='*50}")
    print(f"Original shapes: X_train {X_train.shape}, X_test {X_test.shape}")

    # Flatten 3D → 2D for sklearn
    X_train_flat = flatten_sequences(X_train)
    X_test_flat  = flatten_sequences(X_test)

    print(f"Flattened shapes: X_train {X_train_flat.shape}, X_test {X_test_flat.shape}")

    # Train
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1  # use all CPU cores
    )
    model.fit(X_train_flat, y_train.ravel())
    print(f"Model trained ({n_estimators} trees).")

    # Predict on test set
    y_pred = model.predict(X_test_flat)

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\n--- Test Set Metrics ---")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.6f}")

    return {
        'asset': asset_name,
        'model': 'RandomForest',
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'train_samples': X_train_flat.shape[0],
        'test_samples': X_test_flat.shape[0],
    }


def compare_models(metrics_list):
    """Prints a side-by-side comparison table and declares a winner."""
    df = pd.DataFrame(metrics_list)
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(df[['model', 'rmse', 'mae', 'r2']].to_string(index=False))

    # Determine winner by lowest RMSE
    best = df.loc[df['rmse'].idxmin()]
    worst = df.loc[df['rmse'].idxmax()]
    improvement = ((worst['rmse'] - best['rmse']) / worst['rmse']) * 100

    print(f"\n🏆 Winner: {best['model']}")
    print(f"   RMSE improvement over {worst['model']}: {improvement:.2f}%")

    if best['model'] == 'RandomForest':
        print("   Why: Random Forest captures non-linear patterns in the")
        print("   flattened time-series that Linear Regression cannot model.")
        print("   Its ensemble of decision trees handles feature interactions")
        print("   and noisy financial data more robustly.")
    else:
        print("   Why: Linear Regression's simplicity avoids overfitting on")
        print("   the relatively small dataset. The relationship between")
        print("   flattened features and price is close to linear.")


def run_xgboost(asset_name='Bitcoin', n_estimators=200):
    """
    Trains an XGBoost Regressor baseline on the given asset and
    evaluates RMSE, MAE, R² on the test set.
    """
    prefix = 'btc' if asset_name == 'Bitcoin' else asset_name.lower()

    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_test.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_test.npy"))

    print(f"\n{'='*50}")
    print(f"XGBoost Baseline — {asset_name}")
    print(f"{'='*50}")

    X_train_flat = flatten_sequences(X_train)
    X_test_flat  = flatten_sequences(X_test)

    print(f"Flattened shapes: X_train {X_train_flat.shape}, X_test {X_test_flat.shape}")

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train_flat, y_train.ravel())
    print(f"Model trained ({n_estimators} boosting rounds).")

    y_pred = model.predict(X_test_flat)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\n--- Test Set Metrics ---")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.6f}")

    return {
        'asset': asset_name,
        'model': 'XGBoost',
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'train_samples': X_train_flat.shape[0],
        'test_samples': X_test_flat.shape[0],
    }


def run_gradient_boosting(asset_name='Bitcoin', n_estimators=150):
    """
    Trains Scikit-Learn's native GradientBoostingRegressor baseline.
    This serves as an alternative to XGBoost for native macOS support.
    """
    prefix = 'btc' if asset_name == 'Bitcoin' else asset_name.lower()

    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_test.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_test.npy"))

    print(f"\n{'='*50}")
    print(f"Gradient Boosting Baseline — {asset_name}")
    print(f"{'='*50}")

    X_train_flat = flatten_sequences(X_train)
    X_test_flat  = flatten_sequences(X_test)

    print(f"Flattened shapes: X_train {X_train_flat.shape}, X_test {X_test_flat.shape}")

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train_flat, y_train.ravel())
    print(f"Model trained ({n_estimators} boosting rounds).")

    y_pred = model.predict(X_test_flat)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\n--- Test Set Metrics ---")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.6f}")

    return {
        'asset': asset_name,
        'model': 'GradientBoosting',
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'train_samples': X_train_flat.shape[0],
        'test_samples': X_test_flat.shape[0],
    }


def log_metrics(metrics_list, filepath=None):
    """Saves a list of metric dicts to a CSV file."""
    if filepath is None:
        filepath = os.path.join(RESULTS_DIR, 'baseline_metrics.csv')

    df = pd.DataFrame(metrics_list)
    df.to_csv(filepath, index=False)
    print(f"\nMetrics saved to {filepath}")


if __name__ == "__main__":
    all_metrics = []
    assets = ['Bitcoin', 'Gold', 'Silver']

    for asset in assets:
        lr_metrics = run_linear_regression(asset)
        all_metrics.append(lr_metrics)

        rf_metrics = run_random_forest(asset)
        all_metrics.append(rf_metrics)

        gb_metrics = run_gradient_boosting(asset)
        all_metrics.append(gb_metrics)

    # Compare and log
    compare_models(all_metrics)
    log_metrics(all_metrics)
