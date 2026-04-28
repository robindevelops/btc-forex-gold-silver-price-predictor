"""
Walk-Forward Validation for BTC LSTM.

Implements a rolling window approach:
  - For each test day, take the last seq_len days of ACTUAL scaled data
  - Predict day N+1
  - Slide window forward by 1 day
  - Collect all 1-step-ahead predictions
  - Compute overall metrics + rolling RMSE over time
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import PROCESSED_DATA_DIR, MODELS_DIR, BEST_LSTM_CONFIG
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def inverse_transform_price(scaled_values, scaler, price_col_idx=0, n_features=16):
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, price_col_idx] = np.array(scaled_values).ravel()
    return scaler.inverse_transform(dummy)[:, price_col_idx]


def get_prefix(asset):
    return 'btc' if asset == 'Bitcoin' else asset.lower()


def walk_forward_validate(asset='Bitcoin'):
    """
    Walk-forward validation: for each day in the test period,
    build a fresh input sequence from ACTUAL preceding data and predict 1 step ahead.
    """
    prefix = get_prefix(asset)
    cfg = BEST_LSTM_CONFIG
    seq_len = cfg['seq_len']

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD VALIDATION — {asset}")
    print(f"{'='*70}")

    # ── Load model & scaler ──
    model = load_model(os.path.join(MODELS_DIR, f'{prefix}_lstm_final.keras'))
    scaler = joblib.load(os.path.join(MODELS_DIR, f'{prefix}_scaler.pkl'))

    # ── Load scaled data (continuous) ──
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_train_scaled.csv'),
                           index_col='timestamp', parse_dates=True)
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_val_scaled.csv'),
                         index_col='timestamp', parse_dates=True)
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_test_scaled.csv'),
                          index_col='timestamp', parse_dates=True)

    full_df = pd.concat([train_df, val_df, test_df])
    n_features = full_df.shape[1]
    price_col_idx = list(full_df.columns).index('price')

    test_start_idx = len(train_df) + len(val_df)  # first test row in full_df
    total_test = len(test_df)

    print(f"  Full dataset:  {len(full_df)} rows")
    print(f"  Test period:   {total_test} days ({test_df.index.min().date()} → {test_df.index.max().date()})")
    print(f"  Sequence len:  {seq_len}")
    print(f"  Walk-forward predictions to make: {total_test}")

    # ── Walk forward ──
    predictions_scaled = []
    actuals_scaled = []
    dates = []

    values = full_df.values

    start_time = datetime.now()
    for i in range(total_test):
        # The target day index in full_df
        target_idx = test_start_idx + i

        # Build the input sequence: the seq_len rows BEFORE the target day
        seq_start = target_idx - seq_len
        x_seq = values[seq_start:target_idx]  # shape: (seq_len, n_features)
        x_input = x_seq.reshape(1, seq_len, n_features)

        # Predict
        y_pred = model.predict(x_input, verbose=0)[0, 0]
        y_actual = values[target_idx, price_col_idx]

        predictions_scaled.append(y_pred)
        actuals_scaled.append(y_actual)
        dates.append(full_df.index[target_idx])

        if (i + 1) % 50 == 0 or i == total_test - 1:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"    [{i+1}/{total_test}] {elapsed:.1f}s elapsed")

    # ── Convert to arrays ──
    pred_scaled = np.array(predictions_scaled)
    act_scaled = np.array(actuals_scaled)
    dates = pd.DatetimeIndex(dates)

    # ── Inverse transform to USD ──
    pred_usd = inverse_transform_price(pred_scaled, scaler, price_col_idx, n_features)
    act_usd = inverse_transform_price(act_scaled, scaler, price_col_idx, n_features)

    # ── Overall metrics ──
    rmse = np.sqrt(mean_squared_error(act_usd, pred_usd))
    mae = mean_absolute_error(act_usd, pred_usd)
    r2 = r2_score(act_usd, pred_usd)
    mape = np.mean(np.abs((act_usd - pred_usd) / act_usd)) * 100

    print(f"\n  Walk-Forward Results ({total_test} 1-step-ahead predictions):")
    print(f"    RMSE: ${rmse:,.2f}")
    print(f"    MAE:  ${mae:,.2f}")
    print(f"    MAPE: {mape:.2f}%")
    print(f"    R²:   {r2:.6f}")

    # ── Compute rolling metrics ──
    errors = np.abs(act_usd - pred_usd)
    sq_errors = (act_usd - pred_usd) ** 2

    rolling_window = 20
    rolling_rmse = pd.Series(sq_errors, index=dates).rolling(rolling_window).mean().apply(np.sqrt)
    rolling_mae = pd.Series(errors, index=dates).rolling(rolling_window).mean()

    # ── Plot ──
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # Plot 1: Actual vs Predicted
    axes[0].plot(dates, act_usd, label='Actual', color='#F7931A', linewidth=2)
    axes[0].plot(dates, pred_usd, label='Walk-Forward Predicted', color='#2196F3',
                 linestyle='--', linewidth=1.5)
    axes[0].set_title(f'{asset} Walk-Forward Validation — Actual vs 1-Step-Ahead Predicted',
                      fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price (USD)')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.02, 0.05,
                 f"RMSE=${rmse:,.0f}  |  MAE=${mae:,.0f}  |  MAPE={mape:.1f}%  |  R²={r2:.4f}",
                 transform=axes[0].transAxes, fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Plot 2: Rolling RMSE
    axes[1].plot(rolling_rmse.index, rolling_rmse.values, color='#E91E63', linewidth=2)
    axes[1].axhline(y=rmse, color='gray', linestyle='--', alpha=0.6, label=f'Overall RMSE=${rmse:,.0f}')
    axes[1].set_title(f'{rolling_window}-Day Rolling RMSE Over Time', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Rolling RMSE (USD)')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(rolling_rmse.index, 0, rolling_rmse.values, alpha=0.1, color='#E91E63')

    # Plot 3: Daily prediction errors
    daily_errors = act_usd - pred_usd
    colors = ['#4CAF50' if e >= 0 else '#F44336' for e in daily_errors]
    axes[2].bar(dates, daily_errors, color=colors, alpha=0.7, width=1)
    axes[2].axhline(y=0, color='black', linewidth=0.5)
    axes[2].set_title('Daily Prediction Error (Actual − Predicted)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Error (USD)')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)
    bias = np.mean(daily_errors)
    axes[2].text(0.02, 0.95, f"Mean Bias={bias:+,.0f}  |  Std={np.std(daily_errors):,.0f}",
                 transform=axes[2].transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f'{prefix}_walk_forward.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  ✅ Plot saved to {plot_path}")

    return {
        'asset': asset,
        'n_predictions': total_test,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'mean_bias': bias,
        'plot_path': plot_path,
    }


if __name__ == "__main__":
    results = []
    for asset in ['Bitcoin', 'Gold', 'Silver']:
        r = walk_forward_validate(asset)
        results.append(r)

    # ── Summary ──
    print(f"\n\n{'='*80}")
    print("  WALK-FORWARD VALIDATION — SUMMARY")
    print(f"{'='*80}")
    print(f"\n  {'Asset':<10} {'Days':>6} {'RMSE':>12} {'MAE':>12} {'MAPE':>8} {'R²':>10} {'Bias':>12}")
    print(f"  {'-'*72}")
    for r in results:
        print(f"  {r['asset']:<10} {r['n_predictions']:>6} ${r['rmse']:>10,.2f} ${r['mae']:>10,.2f}"
              f" {r['mape']:>7.2f}% {r['r2']:>9.4f} ${r['mean_bias']:>+10,.0f}")

    df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, 'walk_forward_results.csv')
    df.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}")
