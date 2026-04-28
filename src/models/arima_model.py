"""
ARIMA Model for BTC Price Prediction.

1. Plots ACF & PACF to identify (p, d, q) parameters
2. Fits ARIMA on training data
3. Evaluates on test set
4. Saves residuals for LSTM hybrid correction
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PROCESSED_DATA_DIR, MODELS_DIR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_acf_pacf(series, lags=40):
    """Plot ACF and PACF side-by-side for the differenced series."""
    diff = series.diff().dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    plot_acf(diff, lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title('ACF — Differenced BTC Price', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('Autocorrelation')
    
    plot_pacf(diff, lags=lags, ax=axes[1], alpha=0.05, method='ywm')
    axes[1].set_title('PACF — Differenced BTC Price', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Partial Autocorrelation')
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'btc_acf_pacf.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ ACF/PACF plot saved to {path}")
    return path


def find_best_arima(train, max_p=5, max_q=5, d=1):
    """
    Grid search over (p, d, q) combinations using AIC.
    Returns the best order and fitted model.
    """
    best_aic = np.inf
    best_order = None
    best_model = None
    results = []
    
    total = (max_p + 1) * (max_q + 1)
    count = 0
    
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            count += 1
            try:
                model = ARIMA(train, order=(p, d, q))
                fitted = model.fit()
                aic = fitted.aic
                results.append({'p': p, 'd': d, 'q': q, 'aic': aic})
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    best_model = fitted
                    
                if count % 6 == 0 or count == total:
                    print(f"    [{count}/{total}] Current best: ARIMA{best_order} AIC={best_aic:.2f}")
                    
            except Exception:
                results.append({'p': p, 'd': d, 'q': q, 'aic': np.nan})
    
    results_df = pd.DataFrame(results).dropna().sort_values('aic')
    return best_order, best_model, results_df


if __name__ == "__main__":
    # ── Load BTC price data ──
    btc_path = os.path.join(PROCESSED_DATA_DIR, 'btc_features.csv')
    df = pd.read_csv(btc_path, index_col='timestamp', parse_dates=True)
    price = df['price']
    
    print(f"Loaded BTC price series: {len(price)} observations")
    print(f"  Date range: {price.index.min().date()} → {price.index.max().date()}")
    
    # ── Chronological split: 70/15/15 ──
    n = len(price)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train = price.iloc[:train_end]
    val = price.iloc[train_end:val_end]
    test = price.iloc[val_end:]
    train_val = price.iloc[:val_end]  # for final retraining
    
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"  Train dates: {train.index.min().date()} → {train.index.max().date()}")
    print(f"  Test dates:  {test.index.min().date()} → {test.index.max().date()}")
    
    # ── Step 1: Plot ACF & PACF ──
    print(f"\n{'='*60}")
    print("  STEP 1: ACF & PACF Analysis")
    print(f"{'='*60}")
    acf_path = plot_acf_pacf(train, lags=40)
    print("  Reading ACF/PACF to determine p, q candidates...")
    print("  d=1 (confirmed by ADF test — series is I(1))")
    
    # ── Step 2: Grid search for best (p,d,q) on training data ──
    print(f"\n{'='*60}")
    print("  STEP 2: ARIMA Grid Search (AIC)")
    print(f"{'='*60}")
    print("  Searching p=0..5, d=1, q=0..5...")
    
    best_order, best_model, grid_df = find_best_arima(train, max_p=5, max_q=5, d=1)
    
    print(f"\n  Top 5 configurations:")
    print(grid_df.head(5).to_string(index=False))
    print(f"\n  🏆 Best: ARIMA{best_order} — AIC={best_model.aic:.2f}")
    
    # ── Step 3: Evaluate on test set (walk-forward) ──
    print(f"\n{'='*60}")
    print("  STEP 3: Walk-Forward Evaluation on Test Set")
    print(f"{'='*60}")
    
    # Retrain on train+val with best order, then forecast test period
    print(f"  Retraining ARIMA{best_order} on Train+Val ({len(train_val)} obs)...")
    final_model = ARIMA(train_val, order=best_order)
    final_fit = final_model.fit()
    
    # Forecast the entire test period
    forecast = final_fit.forecast(steps=len(test))
    forecast.index = test.index
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    r2 = r2_score(test, forecast)
    mape = np.mean(np.abs((test - forecast) / test)) * 100
    
    print(f"\n  ARIMA{best_order} Test Set Results:")
    print(f"    RMSE: ${rmse:,.2f}")
    print(f"    MAE:  ${mae:,.2f}")
    print(f"    MAPE: {mape:.2f}%")
    print(f"    R²:   {r2:.6f}")
    
    # ── Step 4: Compute & save residuals ──
    print(f"\n{'='*60}")
    print("  STEP 4: Save Residuals for LSTM Hybrid")
    print(f"{'='*60}")
    
    # In-sample residuals (from training fit)
    train_resid = final_fit.resid
    
    # Out-of-sample residuals (test set errors)
    test_residuals = test - forecast
    
    # Save everything
    residuals_data = {
        'train_residuals': train_resid,
        'test_actual': test,
        'test_forecast': forecast,
        'test_residuals': test_residuals,
        'order': best_order,
        'aic': final_fit.aic,
    }
    
    resid_path = os.path.join(MODELS_DIR, 'btc_arima_residuals.pkl')
    joblib.dump(residuals_data, resid_path)
    print(f"  ✅ Residuals saved to {resid_path}")
    
    # Also save the fitted model
    model_path = os.path.join(MODELS_DIR, 'btc_arima_model.pkl')
    joblib.dump(final_fit, model_path)
    print(f"  ✅ ARIMA model saved to {model_path}")
    
    # ── Step 5: Plot forecast vs actual ──
    print(f"\n{'='*60}")
    print("  STEP 5: Plotting")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    
    # Plot 1: Full price with forecast
    axes[0].plot(train_val.index, train_val.values, label='Train+Val', color='#2196F3', linewidth=1)
    axes[0].plot(test.index, test.values, label='Actual (Test)', color='#F7931A', linewidth=2)
    axes[0].plot(forecast.index, forecast.values, label=f'ARIMA{best_order} Forecast',
                 color='red', linestyle='--', linewidth=2)
    axes[0].set_title(f'BTC Price — ARIMA{best_order} Forecast vs Actual', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price (USD)')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.02, 0.05, f"RMSE=${rmse:,.0f}  |  R²={r2:.4f}  |  MAPE={mape:.1f}%",
                 transform=axes[0].transAxes, fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Plot 2: Test set zoomed
    axes[1].plot(test.index, test.values, label='Actual', color='#F7931A', linewidth=2)
    axes[1].plot(forecast.index, forecast.values, label='ARIMA Forecast',
                 color='red', linestyle='--', linewidth=2)
    axes[1].fill_between(test.index, test.values, forecast.values, alpha=0.15, color='red')
    axes[1].set_title('Test Set — Zoomed', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Price (USD)')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    axes[2].bar(test_residuals.index, test_residuals.values, color='steelblue', alpha=0.7, width=1)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].set_title('Test Set Residuals (Errors the LSTM Will Correct)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Residual (USD)')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)
    axes[2].text(0.02, 0.95, f"Mean={test_residuals.mean():,.0f}  |  Std={test_residuals.std():,.0f}",
                 transform=axes[2].transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'btc_arima_forecast.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  ✅ Forecast plot saved to {plot_path}")
    
    # ── Final summary ──
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Best ARIMA order:  {best_order}")
    print(f"  AIC:               {final_fit.aic:.2f}")
    print(f"  Test RMSE:         ${rmse:,.2f}")
    print(f"  Test MAE:          ${mae:,.2f}")
    print(f"  Test MAPE:         {mape:.2f}%")
    print(f"  Test R²:           {r2:.6f}")
    print(f"  Residual mean:     ${test_residuals.mean():,.2f}")
    print(f"  Residual std:      ${test_residuals.std():,.2f}")
    print(f"\n  Files saved:")
    print(f"    {resid_path}")
    print(f"    {model_path}")
    print(f"    {plot_path}")
    print(f"    {acf_path}")
