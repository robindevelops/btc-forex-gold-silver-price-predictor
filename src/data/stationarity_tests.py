"""
Stationarity Tests for BTC Price Series.

Runs the Augmented Dickey-Fuller (ADF) test on the raw BTC price,
applies first-order differencing if non-stationary, and re-tests.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PROCESSED_DATA_DIR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def run_adf_test(series, series_name="Series"):
    """
    Runs the Augmented Dickey-Fuller test and prints a clear summary.
    
    Null hypothesis (H0): The series has a unit root (non-stationary).
    If p-value < 0.05, we reject H0 → series IS stationary.
    """
    result = adfuller(series.dropna(), autolag='AIC')
    
    adf_stat = result[0]
    p_value = result[1]
    n_lags = result[2]
    n_obs = result[3]
    critical_values = result[4]
    
    print(f"\n{'='*60}")
    print(f"  ADF Test — {series_name}")
    print(f"{'='*60}")
    print(f"  ADF Statistic:   {adf_stat:.6f}")
    print(f"  p-value:         {p_value:.6f}")
    print(f"  Lags Used:       {n_lags}")
    print(f"  Observations:    {n_obs}")
    print(f"  Critical Values:")
    for key, val in critical_values.items():
        marker = " ← REJECTED" if adf_stat < val else ""
        print(f"    {key}: {val:.6f}{marker}")
    
    if p_value < 0.05:
        verdict = "✅ STATIONARY (p < 0.05 — reject H0)"
    else:
        verdict = "❌ NON-STATIONARY (p >= 0.05 — fail to reject H0)"
    
    print(f"\n  Verdict: {verdict}")
    
    return {
        'series': series_name,
        'adf_stat': adf_stat,
        'p_value': p_value,
        'lags': n_lags,
        'observations': n_obs,
        'critical_1pct': critical_values['1%'],
        'critical_5pct': critical_values['5%'],
        'critical_10pct': critical_values['10%'],
        'stationary': p_value < 0.05
    }


if __name__ == "__main__":
    # Load unscaled BTC features (contains the raw 'price' column)
    btc_path = os.path.join(PROCESSED_DATA_DIR, 'btc_features.csv')
    df = pd.read_csv(btc_path, index_col='timestamp', parse_dates=True)
    
    price = df['price']
    print(f"Loaded BTC price series: {len(price)} observations")
    print(f"  Date range: {price.index.min().date()} → {price.index.max().date()}")
    print(f"  Price range: ${price.min():,.2f} → ${price.max():,.2f}")
    
    # --- TEST 1: Raw price series ---
    result_raw = run_adf_test(price, "BTC Raw Price")
    
    # --- Apply first-order differencing ---
    price_diff = price.diff().dropna()
    print(f"\n  Applied first-order differencing.")
    print(f"  Differenced series length: {len(price_diff)}")
    
    # --- TEST 2: Differenced price series ---
    result_diff = run_adf_test(price_diff, "BTC Differenced Price (Δ)")
    
    # --- Plot both series side-by-side ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    axes[0].plot(price.index, price.values, color='#F7931A', linewidth=1.5)
    axes[0].set_title('BTC Raw Price (Non-Stationary)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price (USD)')
    axes[0].grid(True, alpha=0.3)
    stat_text = f"ADF = {result_raw['adf_stat']:.4f}\np = {result_raw['p_value']:.4f}"
    axes[0].text(0.02, 0.95, stat_text, transform=axes[0].transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
    
    axes[1].plot(price_diff.index, price_diff.values, color='#2196F3', linewidth=1, alpha=0.8)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_title('BTC Differenced Price — Δ(Price) (Stationary)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Daily Price Change (USD)')
    axes[1].set_xlabel('Date')
    axes[1].grid(True, alpha=0.3)
    stat_text = f"ADF = {result_diff['adf_stat']:.4f}\np = {result_diff['p_value']:.6f}"
    axes[1].text(0.02, 0.95, stat_text, transform=axes[1].transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))
    
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_path = os.path.join(RESULTS_DIR, 'btc_stationarity_test.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\n  ✅ Plot saved to {plot_path}")
    
    # --- Summary ---
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Raw Price:         {'Stationary' if result_raw['stationary'] else 'Non-Stationary'} (p={result_raw['p_value']:.6f})")
    print(f"  Differenced Price: {'Stationary' if result_diff['stationary'] else 'Non-Stationary'} (p={result_diff['p_value']:.6f})")
    if not result_raw['stationary'] and result_diff['stationary']:
        print(f"\n  ✅ First-order differencing successfully made the series stationary.")
        print(f"     The BTC price series is I(1) — integrated of order 1.")
