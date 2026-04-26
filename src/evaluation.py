"""
Evaluation module for LSTM model predictions.

Loads best model weights, runs predictions on test set,
inverse-transforms using the saved scaler, and computes
RMSE, MAE, R² — then compares vs baseline models.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import MODELS_DIR, PROCESSED_DATA_DIR
from src.model_lstm import build_lstm_model
from src.preprocessing import load_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def inverse_transform_price(scaled_values, scaler, price_col_idx=0, n_features=16):
    """
    Inverse-transform only the price column from MinMaxScaler output.

    The scaler was fit on all 16 features, so we need to create a
    dummy array of the correct shape, fill the price column, inverse
    transform, then extract just the price column.

    Args:
        scaled_values: 1D array of scaled price values.
        scaler: Fitted MinMaxScaler with n_features.
        price_col_idx: Index of the price column (default 0).
        n_features: Total number of features the scaler was fit on.

    Returns:
        1D numpy array of real-scale prices.
    """
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, price_col_idx] = scaled_values.ravel()
    inv = scaler.inverse_transform(dummy)
    return inv[:, price_col_idx]


def evaluate_lstm(asset_name='Bitcoin', checkpoint_path=None):
    """
    Full LSTM evaluation pipeline:
    1. Load best weights
    2. Predict on test set
    3. Inverse-transform to real prices
    4. Compute RMSE, MAE, R²

    Returns:
        dict with model name and metrics (both scaled and real-price).
    """
    prefix = 'btc' if asset_name == 'Bitcoin' else asset_name.lower()

    if checkpoint_path is None:
        checkpoint_path = os.path.join(MODELS_DIR, f'{prefix}_lstm_best.keras')

    # --- Load data ---
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(asset_name)

    print(f"\n{'='*55}")
    print(f"LSTM Evaluation — {asset_name}")
    print(f"{'='*55}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")

    from tensorflow.keras.models import load_model
    model = load_model(checkpoint_path)
    n_features = model.input_shape[2]
    print(f"  Loaded model from: {checkpoint_path}")

    # --- Predict (scaled space) ---
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Metrics in scaled space (for comparison with baseline_metrics.csv)
    rmse_scaled = np.sqrt(mean_squared_error(y_test, y_pred_scaled))
    mae_scaled = mean_absolute_error(y_test, y_pred_scaled)
    r2_scaled = r2_score(y_test, y_pred_scaled)

    print(f"\n--- Scaled-Space Metrics (comparable to baselines) ---")
    print(f"  RMSE: {rmse_scaled:.6f}")
    print(f"  MAE:  {mae_scaled:.6f}")
    print(f"  R²:   {r2_scaled:.6f}")

    # --- Inverse-transform to real prices ---
    scaler_path = os.path.join(MODELS_DIR, f'{prefix}_scaler.pkl')
    scaler = joblib.load(scaler_path)
    print(f"  Loaded scaler from: {scaler_path}")

    y_test_real = inverse_transform_price(y_test, scaler, price_col_idx=0, n_features=n_features)
    y_pred_real = inverse_transform_price(y_pred_scaled, scaler, price_col_idx=0, n_features=n_features)

    # Metrics in real price space
    rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae_real = mean_absolute_error(y_test_real, y_pred_real)
    r2_real = r2_score(y_test_real, y_pred_real)

    print(f"\n--- Real-Price Metrics (USD) ---")
    print(f"  RMSE: ${rmse_real:,.2f}")
    print(f"  MAE:  ${mae_real:,.2f}")
    print(f"  R²:   {r2_real:.6f}")
    print(f"  Price range: ${y_test_real.min():,.2f} — ${y_test_real.max():,.2f}")

    return {
        'asset': asset_name,
        'model': 'LSTM',
        # Scaled metrics (apples-to-apples with baselines)
        'rmse': rmse_scaled,
        'mae': mae_scaled,
        'r2': r2_scaled,
        # Real-price metrics
        'rmse_usd': rmse_real,
        'mae_usd': mae_real,
        'r2_usd': r2_real,
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        # Predictions for plotting
        'y_test_real': y_test_real,
        'y_pred_real': y_pred_real,
    }


def compare_with_baselines(lstm_metrics, asset_name='Bitcoin'):
    """
    Loads baseline_metrics.csv and prints a comparison table
    with the LSTM results.
    """
    baseline_path = os.path.join(RESULTS_DIR, 'baseline_metrics.csv')
    if not os.path.exists(baseline_path):
        print(f"  ⚠ Baseline metrics not found at {baseline_path}")
        return None

    df = pd.read_csv(baseline_path)
    # Filter to the same asset
    df = df[df['asset'] == asset_name][['model', 'rmse', 'mae', 'r2']].copy()

    # Add LSTM row
    lstm_row = pd.DataFrame([{
        'model': 'LSTM',
        'rmse': lstm_metrics['rmse'],
        'mae': lstm_metrics['mae'],
        'r2': lstm_metrics['r2'],
    }])
    df = pd.concat([df, lstm_row], ignore_index=True)

    # Sort by RMSE
    df = df.sort_values('rmse').reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON — {asset_name} (scaled-space metrics)")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    # Determine winner
    best = df.iloc[0]
    print(f"\n🏆 Winner: {best['model']}  (RMSE={best['rmse']:.6f}, R²={best['r2']:.4f})")

    # LSTM vs best baseline
    best_baseline = df[df['model'] != 'LSTM'].iloc[0]
    if lstm_metrics['rmse'] < best_baseline['rmse']:
        improvement = ((best_baseline['rmse'] - lstm_metrics['rmse']) / best_baseline['rmse']) * 100
        print(f"  ✅ LSTM beats best baseline ({best_baseline['model']}) by {improvement:.1f}% RMSE")
    else:
        gap = ((lstm_metrics['rmse'] - best_baseline['rmse']) / best_baseline['rmse']) * 100
        print(f"  ❌ LSTM is {gap:.1f}% worse than best baseline ({best_baseline['model']})")
        print(f"     This is normal for a first run — tuning needed.")

    return df


def plot_predictions(y_test_real, y_pred_real, save_path=None):
    """
    Plot actual vs predicted prices on the test set.
    """
    if save_path is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_path = os.path.join(RESULTS_DIR, 'btc_lstm_predictions.png')

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [3, 1]})

    # --- Top: Actual vs Predicted ---
    ax1 = axes[0]
    x = range(len(y_test_real))
    ax1.plot(x, y_test_real, 'b-', linewidth=1.5, label='Actual Price', alpha=0.9)
    ax1.plot(x, y_pred_real, 'r--', linewidth=1.5, label='LSTM Predicted', alpha=0.8)
    ax1.fill_between(x, y_test_real, y_pred_real, alpha=0.15, color='red')
    ax1.set_title('BTC LSTM — Actual vs Predicted Price (Test Set)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # --- Bottom: Residuals ---
    ax2 = axes[1]
    residuals = y_test_real - y_pred_real
    ax2.bar(x, residuals, color=['green' if r >= 0 else 'red' for r in residuals], alpha=0.6, width=1)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_title('Prediction Residuals (Actual − Predicted)', fontsize=12)
    ax2.set_xlabel('Test Sample Index', fontsize=12)
    ax2.set_ylabel('Residual (USD)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nPrediction plot saved to: {save_path}")


if __name__ == "__main__":
    # --- Evaluate LSTM on BTC test set ---
    lstm_metrics = evaluate_lstm('Bitcoin')

    # --- Compare with baselines ---
    comparison_df = compare_with_baselines(lstm_metrics, 'Bitcoin')

    # --- Plot actual vs predicted ---
    plot_predictions(
        lstm_metrics['y_test_real'],
        lstm_metrics['y_pred_real']
    )
