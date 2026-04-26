"""
Prediction pipeline for BTC LSTM model.

Loads the trained model and scaler, takes the latest data,
and returns next-day price predictions. Also supports
batch prediction and plotting on the test set.
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

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import MODELS_DIR, PROCESSED_DATA_DIR
from src.preprocessing import load_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


class BTCPredictor:
    """
    End-to-end prediction pipeline for BTC price forecasting.

    Loads a trained LSTM model and the fitted MinMaxScaler,
    then provides methods for next-day prediction and
    batch evaluation with visualization.
    """

    def __init__(self, model_path=None, scaler_path=None):
        """
        Initialize the predictor by loading model and scaler.

        Args:
            model_path (str): Path to saved .keras model file.
            scaler_path (str): Path to saved .pkl scaler file.
        """
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, 'btc_lstm_best.keras')
        if scaler_path is None:
            scaler_path = os.path.join(MODELS_DIR, 'btc_scaler.pkl')

        # Load model
        from tensorflow.keras.models import load_model
        self.model = load_model(model_path)
        print(f"✅ Model loaded from: {model_path}")

        # Load scaler
        self.scaler = joblib.load(scaler_path)
        self.n_features = self.scaler.n_features_in_
        self.seq_len = self.model.input_shape[1]
        print(f"✅ Scaler loaded from: {scaler_path}")
        print(f"   Sequence length: {self.seq_len}, Features: {self.n_features}")

    def _inverse_transform_price(self, scaled_values, price_col_idx=0):
        """Inverse-transform scaled price values back to real USD."""
        dummy = np.zeros((len(scaled_values), self.n_features))
        dummy[:, price_col_idx] = np.array(scaled_values).ravel()
        inv = self.scaler.inverse_transform(dummy)
        return inv[:, price_col_idx]

    def predict_next_day(self, latest_sequence):
        """
        Predict the next-day BTC price given a sequence of recent data.

        Args:
            latest_sequence: numpy array of shape (seq_len, n_features),
                             already scaled using the same MinMaxScaler.

        Returns:
            dict with 'scaled_price' and 'price_usd'.
        """
        if latest_sequence.ndim == 2:
            latest_sequence = latest_sequence[np.newaxis, :]  # add batch dim

        assert latest_sequence.shape[1:] == (self.seq_len, self.n_features), \
            f"Expected shape (1, {self.seq_len}, {self.n_features}), got {latest_sequence.shape}"

        # Predict (scaled)
        pred_scaled = self.model.predict(latest_sequence, verbose=0)
        pred_usd = self._inverse_transform_price(pred_scaled)[0]

        return {
            'scaled_price': float(pred_scaled[0, 0]),
            'price_usd': float(pred_usd),
        }

    def predict_from_csv(self, csv_path=None):
        """
        Load the latest scaled data from CSV, extract the last sequence,
        and predict the next-day price.

        Args:
            csv_path (str): Path to a scaled CSV file. Defaults to
                            the BTC test set.

        Returns:
            dict with prediction results and metadata.
        """
        if csv_path is None:
            # Use the most recent available data (test set is the latest)
            csv_path = os.path.join(PROCESSED_DATA_DIR, 'btc_test_scaled.csv')

        df = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)
        print(f"📊 Loaded {len(df)} rows from {os.path.basename(csv_path)}")
        print(f"   Date range: {df.index.min().date()} → {df.index.max().date()}")

        if len(df) < self.seq_len:
            raise ValueError(
                f"Need at least {self.seq_len} rows for prediction, got {len(df)}"
            )

        # Extract last sequence
        latest_seq = df.values[-self.seq_len:]
        result = self.predict_next_day(latest_seq)

        # Add metadata
        last_date = df.index[-1]
        result['prediction_date'] = str(last_date.date() + pd.Timedelta(days=1))
        result['based_on_data_through'] = str(last_date.date())
        result['last_actual_price_scaled'] = float(df['price'].iloc[-1])
        result['last_actual_price_usd'] = float(
            self._inverse_transform_price([df['price'].iloc[-1]])[0]
        )

        return result

    def evaluate_test_set(self):
        """
        Run predictions on the full BTC test set and compute metrics.

        Returns:
            dict with predictions, actuals, and metrics.
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset('Bitcoin')

        # Predict
        y_pred_scaled = self.model.predict(X_test, verbose=0)

        # Scaled metrics
        rmse_s = np.sqrt(mean_squared_error(y_test, y_pred_scaled))
        mae_s = mean_absolute_error(y_test, y_pred_scaled)
        r2_s = r2_score(y_test, y_pred_scaled)

        # Real-price metrics
        y_test_real = self._inverse_transform_price(y_test)
        y_pred_real = self._inverse_transform_price(y_pred_scaled)

        rmse_usd = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        mae_usd = mean_absolute_error(y_test_real, y_pred_real)
        r2_usd = r2_score(y_test_real, y_pred_real)

        print(f"\n{'='*55}")
        print(f"BTC LSTM — Test Set Evaluation")
        print(f"{'='*55}")
        print(f"  Scaled:  RMSE={rmse_s:.6f}  MAE={mae_s:.6f}  R²={r2_s:.6f}")
        print(f"  USD:     RMSE=${rmse_usd:,.2f}  MAE=${mae_usd:,.2f}  R²={r2_usd:.6f}")

        return {
            'y_test_real': y_test_real,
            'y_pred_real': y_pred_real,
            'rmse_scaled': rmse_s,
            'mae_scaled': mae_s,
            'r2_scaled': r2_s,
            'rmse_usd': rmse_usd,
            'mae_usd': mae_usd,
            'r2_usd': r2_usd,
        }

    def plot_actual_vs_predicted(self, y_test_real, y_pred_real, save_path=None):
        """
        Generate a publication-quality Actual vs Predicted plot.
        """
        if save_path is None:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            save_path = os.path.join(RESULTS_DIR, 'btc_actual_vs_predicted.png')

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        mae = mean_absolute_error(y_test_real, y_pred_real)
        r2 = r2_score(y_test_real, y_pred_real)

        fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                                 gridspec_kw={'height_ratios': [3, 1]})

        # ── Top panel: Actual vs Predicted ──
        ax1 = axes[0]
        x = np.arange(len(y_test_real))

        ax1.plot(x, y_test_real, color='#2196F3', linewidth=2,
                 label='Actual Price', zorder=3)
        ax1.plot(x, y_pred_real, color='#FF5722', linewidth=2,
                 linestyle='--', label='LSTM Predicted', zorder=3)
        ax1.fill_between(x, y_test_real, y_pred_real,
                         alpha=0.12, color='#FF5722', zorder=1)

        # Metrics annotation box
        metrics_text = (
            f"RMSE: ${rmse:,.0f}\n"
            f"MAE:  ${mae:,.0f}\n"
            f"R²:   {r2:.4f}"
        )
        ax1.text(0.02, 0.97, metrics_text, transform=ax1.transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                           edgecolor='gray', alpha=0.9),
                 fontfamily='monospace')

        ax1.set_title('BTC Price — Actual vs LSTM Predicted (Test Set)',
                      fontsize=15, fontweight='bold', pad=12)
        ax1.set_ylabel('Price (USD)', fontsize=12)
        ax1.legend(fontsize=12, loc='upper right')
        ax1.grid(True, alpha=0.25)
        ax1.set_xlim(0, len(y_test_real) - 1)

        # Format y-axis with $ and commas
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))

        # ── Bottom panel: Residuals ──
        ax2 = axes[1]
        residuals = y_test_real - y_pred_real
        colors = ['#4CAF50' if r >= 0 else '#f44336' for r in residuals]
        ax2.bar(x, residuals, color=colors, alpha=0.7, width=1, edgecolor='none')
        ax2.axhline(y=0, color='black', linewidth=0.8)

        # Mean residual line
        mean_res = np.mean(residuals)
        ax2.axhline(y=mean_res, color='orange', linewidth=1.2, linestyle='--',
                     label=f'Mean residual: ${mean_res:,.0f}')

        ax2.set_title('Prediction Residuals (Actual − Predicted)', fontsize=12)
        ax2.set_xlabel('Test Sample Index', fontsize=12)
        ax2.set_ylabel('Residual (USD)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.25)
        ax2.set_xlim(0, len(y_test_real) - 1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n📈 Plot saved to: {save_path}")
        return save_path


def predict_next_day_price():
    """
    Convenience function: loads model, predicts next-day BTC price
    from the latest available data.
    """
    predictor = BTCPredictor()
    result = predictor.predict_from_csv()

    print(f"\n{'='*50}")
    print(f"🔮 BTC Next-Day Price Prediction")
    print(f"{'='*50}")
    print(f"  Based on data through: {result['based_on_data_through']}")
    print(f"  Prediction for:        {result['prediction_date']}")
    print(f"  Last actual price:     ${result['last_actual_price_usd']:,.2f}")
    print(f"  Predicted price:       ${result['price_usd']:,.2f}")

    delta = result['price_usd'] - result['last_actual_price_usd']
    pct = (delta / result['last_actual_price_usd']) * 100
    direction = "📈 UP" if delta > 0 else "📉 DOWN"
    print(f"  Change:                {direction} ${abs(delta):,.2f} ({pct:+.2f}%)")
    print(f"{'='*50}")

    return result


if __name__ == "__main__":
    # --- Next-day prediction ---
    result = predict_next_day_price()

    # --- Full test set evaluation + plot ---
    predictor = BTCPredictor()
    eval_results = predictor.evaluate_test_set()

    plot_path = predictor.plot_actual_vs_predicted(
        eval_results['y_test_real'],
        eval_results['y_pred_real']
    )
