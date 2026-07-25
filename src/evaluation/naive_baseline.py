"""
Naive Baseline Evaluation.

A "Naive" model in financial time series assumes that the best predictor
of tomorrow's price is today's price (i.e. predicting zero log return).
This is the hardest baseline to beat for directional metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import PROCESSED_DATA_DIR, RESULTS_DIR

def compute_metrics(y_true, y_pred, model_name=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n--- {model_name} Metrics ---")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    
    return rmse, mae, mape, r2

def evaluate_naive_baseline(asset="Bitcoin"):
    """
    Evaluates the naive forecast (tomorrow's price = today's price)
    on the test set.
    """
    prefix = 'btc' if asset == 'Bitcoin' else asset.lower()
    features_path = os.path.join(PROCESSED_DATA_DIR, f'{prefix}_features.csv')
    
    if not os.path.exists(features_path):
        print(f"Features file for {asset} not found.")
        return
        
    df = pd.read_csv(features_path, index_col='timestamp', parse_dates=True)
    price = df['price']
    
    # Same split as the rest of the pipeline (Test is last 15%)
    n = len(price)
    val_end = int(n * 0.85)
    
    # Test actuals
    test_actual = price.iloc[val_end:].values
    # Previous day's price (the naive forecast for today)
    naive_forecast = price.iloc[val_end - 1:-1].values
    
    print(f"\nEvaluating Naive Baseline for {asset}")
    print(f"Test period size: {len(test_actual)} days")
    
    compute_metrics(test_actual, naive_forecast, f"Naive Baseline ({asset})")
    
    # Try to load LSTM results to compare
    comp_file = os.path.join(RESULTS_DIR, 'final_performance_table.csv')
    if os.path.exists(comp_file):
        comp_df = pd.read_csv(comp_file)
        lstm_row = comp_df[(comp_df['Asset'] == asset) & (comp_df['Model'].str.contains('LSTM'))]
        if not lstm_row.empty:
            lstm_rmse = float(lstm_row['RMSE'].iloc[0].replace('$', '').replace(',', ''))
            naive_rmse = np.sqrt(mean_squared_error(test_actual, naive_forecast))
            diff = naive_rmse - lstm_rmse
            pct = (diff / naive_rmse) * 100
            
            print(f"\n--- Comparison vs LSTM ---")
            print(f"  LSTM RMSE:  ${lstm_rmse:,.2f}")
            print(f"  Naive RMSE: ${naive_rmse:,.2f}")
            if diff > 0:
                print(f"  LSTM beats Naive by ${diff:,.2f} ({pct:.1f}%)")
            else:
                print(f"  Naive beats LSTM by ${-diff:,.2f} ({-pct:.1f}%)")

if __name__ == "__main__":
    for asset in ["Bitcoin", "Gold", "Silver"]:
        evaluate_naive_baseline(asset)
