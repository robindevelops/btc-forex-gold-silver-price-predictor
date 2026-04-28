"""
Comprehensive Backtesting & Final Metrics Table.

Compares ALL models across ALL assets with:
  - RMSE, MAE, MAPE, R²
  - Directional Accuracy (% correct up/down calls)
  - Naive Forecast baseline (tomorrow = today)
  - Buy-and-Hold return comparison

Produces the final results table for the report.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import PROCESSED_DATA_DIR, MODELS_DIR, BEST_LSTM_CONFIG
from src.preprocessing import create_sequences

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════

def get_prefix(asset):
    return 'btc' if asset == 'Bitcoin' else asset.lower()

def inverse_transform_price(scaled_values, scaler, price_col_idx=0, n_features=16):
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, price_col_idx] = np.array(scaled_values).ravel()
    return scaler.inverse_transform(dummy)[:, price_col_idx]

def directional_accuracy(y_true, y_pred, y_prev):
    """% of days where the model correctly predicted the direction (up/down)."""
    actual_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    correct = np.sum(actual_dir == pred_dir)
    return correct / len(y_true) * 100

def compute_all_metrics(y_true, y_pred, y_prev=None):
    """Compute RMSE, MAE, MAPE, R², and optionally Directional Accuracy."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    result = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}
    
    if y_prev is not None:
        result['Dir_Acc'] = directional_accuracy(y_true, y_pred, y_prev)
    else:
        result['Dir_Acc'] = np.nan
        
    return result


# ═══════════════════════════════════════════════════════════
#  MODEL EVALUATORS
# ═══════════════════════════════════════════════════════════

def get_test_prices(asset):
    """Load raw test prices and the preceding day for directional accuracy."""
    prefix = get_prefix(asset)
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_features.csv'),
                     index_col='timestamp', parse_dates=True)
    price = df['price']
    n = len(price)
    val_end = int(n * 0.85)
    
    test = price.iloc[val_end:].values
    prev = price.iloc[val_end - 1:-1].values  # the day before each test day
    dates = price.iloc[val_end:].index
    return test, prev, dates


def eval_naive(asset):
    """Naive forecast: tomorrow's price = today's price."""
    test, prev, _ = get_test_prices(asset)
    # Naive prediction: use yesterday's actual price as today's forecast
    y_pred = prev  # prev[i] is the day before test[i]
    metrics = compute_all_metrics(test, y_pred, prev)
    metrics['Model'] = 'Naive (t=t-1)'
    metrics['Asset'] = asset
    return metrics


def eval_lstm(asset):
    """Load final LSTM, walk-forward 1-step-ahead on test set."""
    prefix = get_prefix(asset)
    cfg = BEST_LSTM_CONFIG
    seq_len = cfg['seq_len']
    
    model = load_model(os.path.join(MODELS_DIR, f'{prefix}_lstm_final.keras'))
    scaler = joblib.load(os.path.join(MODELS_DIR, f'{prefix}_scaler.pkl'))
    
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_train_scaled.csv'),
                           index_col='timestamp', parse_dates=True)
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_val_scaled.csv'),
                         index_col='timestamp', parse_dates=True)
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_test_scaled.csv'),
                          index_col='timestamp', parse_dates=True)
    
    full_df = pd.concat([train_df, val_df, test_df])
    n_features = full_df.shape[1]
    price_col_idx = list(full_df.columns).index('price')
    values = full_df.values
    test_start = len(train_df) + len(val_df)
    total_test = len(test_df)
    
    preds_scaled = []
    for i in range(total_test):
        target_idx = test_start + i
        x_seq = values[target_idx - seq_len:target_idx].reshape(1, seq_len, n_features)
        preds_scaled.append(model.predict(x_seq, verbose=0)[0, 0])
    
    pred_usd = inverse_transform_price(np.array(preds_scaled), scaler, price_col_idx, n_features)
    
    test_actual, prev, _ = get_test_prices(asset)
    metrics = compute_all_metrics(test_actual, pred_usd, prev)
    metrics['Model'] = 'LSTM (Optimized)'
    metrics['Asset'] = asset
    return metrics, pred_usd


def eval_arima(asset):
    """Fit ARIMA(1,1,0), forecast test period."""
    prefix = get_prefix(asset)
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_features.csv'),
                     index_col='timestamp', parse_dates=True)
    price = df['price']
    n = len(price)
    val_end = int(n * 0.85)
    
    train_val = price.iloc[:val_end]
    test = price.iloc[val_end:]
    
    model = ARIMA(train_val, order=(1, 1, 0))
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test)).values
    
    test_actual, prev, _ = get_test_prices(asset)
    metrics = compute_all_metrics(test_actual, forecast, prev)
    metrics['Model'] = 'ARIMA(1,1,0)'
    metrics['Asset'] = asset
    return metrics, forecast


def eval_baseline(asset, model_type):
    """Train and evaluate sklearn baseline."""
    prefix = get_prefix(asset)
    scaler = joblib.load(os.path.join(MODELS_DIR, f'{prefix}_scaler.pkl'))
    
    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_X_train.npy'))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_y_train.npy'))
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_X_test.npy'))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_y_test.npy'))
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    if model_type == 'LinearRegression':
        model = LinearRegression()
        label = 'Linear Regression'
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        label = 'Random Forest'
    
    model.fit(X_train_flat, y_train.ravel())
    y_pred = model.predict(X_test_flat).reshape(-1, 1)
    
    n_features = X_train.shape[2]
    y_test_real = inverse_transform_price(y_test, scaler, 0, n_features)
    y_pred_real = inverse_transform_price(y_pred, scaler, 0, n_features)
    
    # For directional accuracy we need "previous day" in USD
    # The baselines use the old seq=60 .npy files, different test set alignment
    # We'll compute directional accuracy relative to the first value in each sequence
    # (last price the model "saw")
    last_seen_scaled = X_test[:, -1, 0]  # last timestep, price column
    last_seen_real = inverse_transform_price(last_seen_scaled, scaler, 0, n_features)
    
    metrics = compute_all_metrics(y_test_real, y_pred_real, last_seen_real)
    metrics['Model'] = label
    metrics['Asset'] = asset
    return metrics


def eval_ensemble(asset, lstm_pred, arima_pred):
    """Weighted ensemble: 0.3*ARIMA + 0.7*LSTM (aligned on matching length)."""
    test_actual, prev, _ = get_test_prices(asset)
    
    min_len = min(len(lstm_pred), len(arima_pred), len(test_actual))
    a = test_actual[:min_len]
    p = prev[:min_len]
    ens = 0.3 * arima_pred[:min_len] + 0.7 * lstm_pred[:min_len]
    
    metrics = compute_all_metrics(a, ens, p)
    metrics['Model'] = 'Ensemble (0.3A+0.7L)'
    metrics['Asset'] = asset
    return metrics


def eval_buy_and_hold(asset):
    """Buy-and-hold return over the test period."""
    test, _, dates = get_test_prices(asset)
    bnh_return = (test[-1] - test[0]) / test[0] * 100
    return bnh_return, test[0], test[-1]


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_results = []
    bnh_results = []
    
    for asset in ['Bitcoin', 'Gold', 'Silver']:
        print(f"\n{'='*60}")
        print(f"  Evaluating {asset}...")
        print(f"{'='*60}")
        
        # Naive
        print(f"  -> Naive forecast")
        all_results.append(eval_naive(asset))
        
        # LSTM
        print(f"  -> LSTM (walk-forward)")
        lstm_metrics, lstm_pred = eval_lstm(asset)
        all_results.append(lstm_metrics)
        
        # ARIMA
        print(f"  -> ARIMA(1,1,0)")
        arima_metrics, arima_pred = eval_arima(asset)
        all_results.append(arima_metrics)
        
        # Ensemble
        print(f"  -> Ensemble")
        all_results.append(eval_ensemble(asset, lstm_pred, arima_pred))
        
        # Baselines
        print(f"  -> Linear Regression")
        all_results.append(eval_baseline(asset, 'LinearRegression'))
        
        print(f"  -> Random Forest")
        all_results.append(eval_baseline(asset, 'RandomForest'))
        
        # Buy-and-Hold
        bnh_ret, start_p, end_p = eval_buy_and_hold(asset)
        bnh_results.append({'Asset': asset, 'Start': start_p, 'End': end_p, 'Return': bnh_ret})
    
    # ═══════════════════════════════════════════════════════════
    #  BUILD FINAL TABLE
    # ═══════════════════════════════════════════════════════════
    
    df = pd.DataFrame(all_results)
    
    # Reorder columns
    df = df[['Asset', 'Model', 'RMSE', 'MAE', 'MAPE', 'R2', 'Dir_Acc']]
    
    # Sort: by asset, then RMSE
    df = df.sort_values(['Asset', 'RMSE'])
    
    print(f"\n\n{'='*100}")
    print("  FINAL PERFORMANCE TABLE — ALL MODELS × ALL METRICS")
    print(f"{'='*100}")
    
    for asset in ['Bitcoin', 'Gold', 'Silver']:
        adf = df[df['Asset'] == asset].copy()
        print(f"\n  ┌─── {asset.upper()} {'─'*80}")
        
        fmt = adf.copy()
        fmt['RMSE'] = fmt['RMSE'].apply(lambda x: f"${x:,.2f}")
        fmt['MAE'] = fmt['MAE'].apply(lambda x: f"${x:,.2f}")
        fmt['MAPE'] = fmt['MAPE'].apply(lambda x: f"{x:.2f}%")
        fmt['R2'] = fmt['R2'].apply(lambda x: f"{x:.4f}")
        fmt['Dir_Acc'] = fmt['Dir_Acc'].apply(lambda x: f"{x:.1f}%" if not np.isnan(x) else "N/A")
        
        print(fmt[['Model', 'RMSE', 'MAE', 'MAPE', 'R2', 'Dir_Acc']].to_string(index=False))
        
        # Show buy-and-hold for context
        bnh = [b for b in bnh_results if b['Asset'] == asset][0]
        print(f"  └─── Buy & Hold: ${bnh['Start']:,.2f} → ${bnh['End']:,.2f} ({bnh['Return']:+.2f}%)")
    
    # ═══════════════════════════════════════════════════════════
    #  DIRECTIONAL ACCURACY COMPARISON
    # ═══════════════════════════════════════════════════════════
    
    print(f"\n\n{'='*80}")
    print("  DIRECTIONAL ACCURACY (% Correct Up/Down Calls)")
    print(f"{'='*80}")
    
    dir_df = df[df['Dir_Acc'].notna()][['Asset', 'Model', 'Dir_Acc']].copy()
    
    for asset in ['Bitcoin', 'Gold', 'Silver']:
        adf = dir_df[dir_df['Asset'] == asset].sort_values('Dir_Acc', ascending=False)
        print(f"\n  {asset}:")
        for _, row in adf.iterrows():
            bar = '█' * int(row['Dir_Acc'] / 2)
            print(f"    {row['Model']:<25} {row['Dir_Acc']:5.1f}% {bar}")
    
    # ═══════════════════════════════════════════════════════════
    #  SAVE
    # ═══════════════════════════════════════════════════════════
    
    out_path = os.path.join(RESULTS_DIR, 'final_performance_table.csv')
    df.to_csv(out_path, index=False)
    print(f"\n\n  ✅ Saved final table to {out_path}")
    
    bnh_path = os.path.join(RESULTS_DIR, 'buy_and_hold.csv')
    pd.DataFrame(bnh_results).to_csv(bnh_path, index=False)
    print(f"  ✅ Saved buy-and-hold to {bnh_path}")
    
    # ═══════════════════════════════════════════════════════════
    #  PLOT: Directional Accuracy bar chart
    # ═══════════════════════════════════════════════════════════
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    colors = {'LSTM (Optimized)': '#2196F3', 'ARIMA(1,1,0)': '#FF9800',
              'Ensemble (0.3A+0.7L)': '#E91E63', 'Naive (t=t-1)': '#9E9E9E',
              'Linear Regression': '#4CAF50', 'Random Forest': '#795548'}
    
    for idx, asset in enumerate(['Bitcoin', 'Gold', 'Silver']):
        adf = dir_df[dir_df['Asset'] == asset].sort_values('Dir_Acc', ascending=True)
        bars = axes[idx].barh(adf['Model'], adf['Dir_Acc'],
                              color=[colors.get(m, '#607D8B') for m in adf['Model']])
        axes[idx].set_title(asset, fontsize=14, fontweight='bold')
        axes[idx].set_xlim(0, 100)
        axes[idx].axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
        axes[idx].grid(True, alpha=0.2, axis='x')
        
        for bar, val in zip(bars, adf['Dir_Acc']):
            axes[idx].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                          f'{val:.1f}%', va='center', fontsize=10)
    
    axes[0].set_xlabel('Directional Accuracy (%)')
    plt.suptitle('Directional Accuracy — % Correct Up/Down Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    dir_plot = os.path.join(RESULTS_DIR, 'directional_accuracy.png')
    plt.savefig(dir_plot, dpi=150)
    plt.close()
    print(f"  ✅ Saved directional accuracy plot to {dir_plot}")
