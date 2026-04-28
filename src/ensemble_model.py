"""
ARIMA + LSTM Hybrid Ensemble.

For each asset (BTC, Gold, Silver):
  1. Fit ARIMA on raw prices (train+val), forecast test period
  2. Load LSTM final model, predict on test sequences (inverse-transform to USD)
  3. Align predictions on matching dates
  4. Combine: ensemble = 0.3 × ARIMA + 0.7 × LSTM
  5. Evaluate ensemble vs standalone LSTM vs standalone ARIMA
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import PROCESSED_DATA_DIR, MODELS_DIR, BEST_LSTM_CONFIG
from src.preprocessing import create_sequences

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_prefix(asset):
    return 'btc' if asset == 'Bitcoin' else asset.lower()


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'RMSE_USD': rmse, 'MAE_USD': mae, 'R2': r2, 'MAPE': mape}


def inverse_transform_price(scaled_values, scaler, price_col_idx=0, n_features=16):
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, price_col_idx] = np.array(scaled_values).ravel()
    return scaler.inverse_transform(dummy)[:, price_col_idx]


def get_arima_forecast(asset):
    """Fit ARIMA(1,1,0) on train+val, forecast the test period. Returns (test_actual, forecast) as Series."""
    prefix = get_prefix(asset)
    features_path = os.path.join(PROCESSED_DATA_DIR, f'{prefix}_features.csv')
    df = pd.read_csv(features_path, index_col='timestamp', parse_dates=True)
    price = df['price']

    n = len(price)
    val_end = int(n * 0.85)

    train_val = price.iloc[:val_end]
    test = price.iloc[val_end:]

    # Fit ARIMA(1,1,0) — best order found for BTC, reasonable starting point for all
    model = ARIMA(train_val, order=(1, 1, 0))
    fitted = model.fit()

    forecast = fitted.forecast(steps=len(test))
    forecast.index = test.index

    return test, forecast


def get_lstm_predictions(asset):
    """Load final LSTM, predict on test set, inverse-transform to USD. Returns (dates, actual_usd, pred_usd)."""
    prefix = get_prefix(asset)
    seq_len = BEST_LSTM_CONFIG['seq_len']

    model_path = os.path.join(MODELS_DIR, f'{prefix}_lstm_final.keras')
    scaler_path = os.path.join(MODELS_DIR, f'{prefix}_scaler.pkl')

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_train_scaled.csv'),
                           index_col='timestamp', parse_dates=True)
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_val_scaled.csv'),
                         index_col='timestamp', parse_dates=True)
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_test_scaled.csv'),
                          index_col='timestamp', parse_dates=True)

    L_train = len(train_df)
    L_val = len(val_df)

    full_df = pd.concat([train_df, val_df, test_df])
    X, y = create_sequences(full_df, seq_len=seq_len)

    split_2 = L_train + L_val - seq_len
    X_test, y_test = X[split_2:], y[split_2:]

    # The date for each prediction is the day AFTER the last day in the sequence
    test_dates = full_df.index[L_train + L_val:]

    y_pred_scaled = model.predict(X_test, verbose=0)
    n_features = X_test.shape[2]

    y_test_real = inverse_transform_price(y_test, scaler, 0, n_features)
    y_pred_real = inverse_transform_price(y_pred_scaled, scaler, 0, n_features)

    return test_dates, y_test_real, y_pred_real


def run_ensemble(asset, arima_weight=0.3, lstm_weight=0.7):
    """Run the full ensemble pipeline for one asset."""
    prefix = get_prefix(asset)
    print(f"\n{'='*70}")
    print(f"  {asset.upper()} — ARIMA+LSTM Ensemble ({arima_weight:.0%} ARIMA + {lstm_weight:.0%} LSTM)")
    print(f"{'='*70}")

    # ── Get ARIMA forecast ──
    print("  Fitting ARIMA(1,1,0)...")
    arima_actual, arima_forecast = get_arima_forecast(asset)
    print(f"    ARIMA test range: {arima_actual.index.min().date()} → {arima_actual.index.max().date()} ({len(arima_actual)} days)")

    # ── Get LSTM predictions ──
    print("  Loading LSTM final model...")
    lstm_dates, lstm_actual, lstm_pred = get_lstm_predictions(asset)
    print(f"    LSTM test range: {lstm_dates.min().date()} → {lstm_dates.max().date()} ({len(lstm_dates)} days)")

    # ── Align on matching dates ──
    arima_df = pd.DataFrame({'arima_actual': arima_actual, 'arima_pred': arima_forecast})
    lstm_df = pd.DataFrame({'lstm_actual': lstm_actual, 'lstm_pred': lstm_pred}, index=lstm_dates)

    merged = arima_df.join(lstm_df, how='inner')
    print(f"    Aligned: {len(merged)} matching test days")

    actual = merged['arima_actual'].values
    arima_pred = merged['arima_pred'].values
    lstm_pred_vals = merged['lstm_pred'].values
    ensemble_pred = arima_weight * arima_pred + lstm_weight * lstm_pred_vals
    dates = merged.index

    # ── Evaluate all three ──
    m_arima = compute_metrics(actual, arima_pred)
    m_lstm = compute_metrics(actual, lstm_pred_vals)
    m_ensemble = compute_metrics(actual, ensemble_pred)

    print(f"\n  {'Model':<25} {'RMSE':>12} {'MAE':>12} {'MAPE':>8} {'R²':>10}")
    print(f"  {'-'*67}")
    for name, m in [('ARIMA(1,1,0)', m_arima), ('LSTM (Optimized)', m_lstm), ('Ensemble (0.3+0.7)', m_ensemble)]:
        print(f"  {name:<25} ${m['RMSE_USD']:>10,.2f} ${m['MAE_USD']:>10,.2f} {m['MAPE']:>7.2f}% {m['R2']:>9.4f}")

    # ── Winner ──
    if m_ensemble['RMSE_USD'] < m_lstm['RMSE_USD']:
        improvement = (1 - m_ensemble['RMSE_USD'] / m_lstm['RMSE_USD']) * 100
        print(f"\n  ✅ Ensemble WINS — {improvement:.2f}% RMSE improvement over standalone LSTM")
        winner = 'Ensemble'
    else:
        degradation = (m_ensemble['RMSE_USD'] / m_lstm['RMSE_USD'] - 1) * 100
        print(f"\n  ❌ Ensemble LOSES — {degradation:.2f}% RMSE degradation vs standalone LSTM")
        winner = 'LSTM'

    # ── Plot ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    axes[0].plot(dates, actual, label='Actual', color='#F7931A', linewidth=2)
    axes[0].plot(dates, arima_pred, label='ARIMA', color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[0].plot(dates, lstm_pred_vals, label='LSTM', color='#2196F3', linestyle='--', linewidth=1.5)
    axes[0].plot(dates, ensemble_pred, label='Ensemble (0.3A+0.7L)', color='#E91E63', linewidth=2)
    axes[0].set_title(f'{asset} — Ensemble vs Standalone Models', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price (USD)')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Error bars
    err_lstm = np.abs(actual - lstm_pred_vals)
    err_ensemble = np.abs(actual - ensemble_pred)
    axes[1].plot(dates, err_lstm, label=f'LSTM Error (MAE=${m_lstm["MAE_USD"]:,.0f})',
                 color='#2196F3', linewidth=1.5, alpha=0.7)
    axes[1].plot(dates, err_ensemble, label=f'Ensemble Error (MAE=${m_ensemble["MAE_USD"]:,.0f})',
                 color='#E91E63', linewidth=1.5, alpha=0.7)
    axes[1].set_title('Absolute Prediction Error Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Absolute Error (USD)')
    axes[1].set_xlabel('Date')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f'{prefix}_ensemble_comparison.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  ✅ Plot saved to {plot_path}")

    return {
        'Asset': asset,
        'ARIMA_RMSE': m_arima['RMSE_USD'],
        'ARIMA_R2': m_arima['R2'],
        'LSTM_RMSE': m_lstm['RMSE_USD'],
        'LSTM_MAE': m_lstm['MAE_USD'],
        'LSTM_MAPE': m_lstm['MAPE'],
        'LSTM_R2': m_lstm['R2'],
        'Ensemble_RMSE': m_ensemble['RMSE_USD'],
        'Ensemble_MAE': m_ensemble['MAE_USD'],
        'Ensemble_MAPE': m_ensemble['MAPE'],
        'Ensemble_R2': m_ensemble['R2'],
        'Winner': winner,
        'plot_path': plot_path,
    }


if __name__ == "__main__":
    all_results = []

    for asset in ['Bitcoin', 'Gold', 'Silver']:
        result = run_ensemble(asset)
        all_results.append(result)

    # ── Final comparison table ──
    print(f"\n\n{'='*80}")
    print("  FINAL ENSEMBLE COMPARISON — All Assets")
    print(f"{'='*80}")

    df = pd.DataFrame(all_results)
    print(f"\n  {'Asset':<10} {'LSTM RMSE':>12} {'Ensemble RMSE':>15} {'Δ':>8} {'Winner':>10}")
    print(f"  {'-'*60}")
    for _, row in df.iterrows():
        delta = ((row['Ensemble_RMSE'] / row['LSTM_RMSE']) - 1) * 100
        sign = '+' if delta > 0 else ''
        print(f"  {row['Asset']:<10} ${row['LSTM_RMSE']:>10,.2f} ${row['Ensemble_RMSE']:>13,.2f} {sign}{delta:>6.2f}% {row['Winner']:>10}")

    # Save results
    out_path = os.path.join(RESULTS_DIR, 'ensemble_comparison.csv')
    df.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}")
