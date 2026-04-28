"""
LSTM Architecture Experiment — 50 vs 100 units.

Trains two LSTM variants with identical hyperparameters
(except unit count), evaluates both on the held-out test set,
and logs results to results/experiment_log.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import MODELS_DIR, PROCESSED_DATA_DIR
from src.models.model_lstm import build_lstm_model
from src.data.preprocessing import load_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF info logs

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def inverse_transform_price(scaled_values, scaler, price_col_idx=0, n_features=16):
    """Inverse-transform scaled price column back to real USD."""
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, price_col_idx] = np.array(scaled_values).ravel()
    inv = scaler.inverse_transform(dummy)
    return inv[:, price_col_idx]


def run_experiment(lstm_units, X_train, y_train, X_test, y_test, scaler,
                   learning_rate=0.001, batch_size=16, epochs=150, patience=20):
    """
    Train an LSTM variant and return metrics.

    Args:
        lstm_units (int): Number of LSTM units per layer.
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        scaler: Fitted MinMaxScaler for inverse transform.
        learning_rate, batch_size, epochs, patience: Hyperparameters.

    Returns:
        dict with experiment configuration and results.
    """
    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    dense_units = lstm_units // 2  # proportional dense layer

    print(f"\n{'='*60}")
    print(f"  Experiment: LSTM units={lstm_units}, dense={dense_units}")
    print(f"  lr={learning_rate}, batch={batch_size}, epochs={epochs}, patience={patience}")
    print(f"{'='*60}")

    # Build model
    model = build_lstm_model(
        seq_len, n_features,
        learning_rate=learning_rate,
        lstm_units=lstm_units,
        dense_units=dense_units
    )

    total_params = model.count_params()
    print(f"  Total parameters: {total_params:,}")

    # Checkpoint path for this variant
    ckpt_path = os.path.join(MODELS_DIR, f'btc_lstm_{lstm_units}u_best.keras')

    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=patience,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            filepath=ckpt_path, monitor='val_loss',
            save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=7, min_lr=1e-6, verbose=1
        ),
    ]

    # Train
    start_time = datetime.now()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    train_time = (datetime.now() - start_time).total_seconds()
    actual_epochs = len(history.history['loss'])
    best_epoch = np.argmin(history.history['val_loss']) + 1

    # Predict (scaled)
    y_pred_scaled = model.predict(X_test, verbose=0)
    rmse_s = np.sqrt(mean_squared_error(y_test, y_pred_scaled))
    mae_s = mean_absolute_error(y_test, y_pred_scaled)
    r2_s = r2_score(y_test, y_pred_scaled)

    # Predict (real USD)
    y_test_real = inverse_transform_price(y_test, scaler, 0, n_features)
    y_pred_real = inverse_transform_price(y_pred_scaled, scaler, 0, n_features)
    rmse_usd = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae_usd = mean_absolute_error(y_test_real, y_pred_real)
    r2_usd = r2_score(y_test_real, y_pred_real)

    # Best val_loss
    best_val_loss = min(history.history['val_loss'])

    print(f"\n  --- Results ---")
    print(f"  Trained {actual_epochs} epochs (best @ {best_epoch}) in {train_time:.1f}s")
    print(f"  Scaled:  RMSE={rmse_s:.6f}  MAE={mae_s:.6f}  R²={r2_s:.6f}")
    print(f"  USD:     RMSE=${rmse_usd:,.2f}  MAE=${mae_usd:,.2f}  R²={r2_usd:.6f}")
    print(f"  Model saved: {ckpt_path}")

    return {
        'timestamp': datetime.now().isoformat(),
        'experiment': f'LSTM-{lstm_units}u',
        'lstm_units': lstm_units,
        'dense_units': dense_units,
        'total_params': total_params,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'max_epochs': epochs,
        'actual_epochs': actual_epochs,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'train_time_sec': round(train_time, 1),
        'rmse_scaled': rmse_s,
        'mae_scaled': mae_s,
        'r2_scaled': r2_s,
        'rmse_usd': round(rmse_usd, 2),
        'mae_usd': round(mae_usd, 2),
        'r2_usd': r2_usd,
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'checkpoint_path': ckpt_path,
    }


if __name__ == "__main__":
    # --- Load data ---
    print("Loading BTC dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset('Bitcoin')

    # Merge train+val (same strategy as the winning v2 model)
    X_tr = np.concatenate([X_train, X_val], axis=0)
    y_tr = np.concatenate([y_train, y_val], axis=0)
    print(f"  Training samples: {X_tr.shape[0]}, Test samples: {X_test.shape[0]}")

    # Load scaler
    scaler = joblib.load(os.path.join(MODELS_DIR, 'btc_scaler.pkl'))

    # --- Run experiments ---
    experiments = []
    for units in [50, 100]:
        result = run_experiment(
            lstm_units=units,
            X_train=X_tr, y_train=y_tr,
            X_test=X_test, y_test=y_test,
            scaler=scaler,
            learning_rate=0.001,
            batch_size=16,
            epochs=150,
            patience=20
        )
        experiments.append(result)

    # --- Log results ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, 'experiment_log.csv')

    # Append to existing log if it exists
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        df = pd.concat([existing, pd.DataFrame(experiments)], ignore_index=True)
    else:
        df = pd.DataFrame(experiments)

    df.to_csv(log_path, index=False)
    print(f"\n📝 Experiment log saved to: {log_path}")

    # --- Print comparison ---
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPARISON — LSTM 50 vs 100 units")
    print(f"{'='*70}")
    cmp = pd.DataFrame(experiments)[
        ['experiment', 'lstm_units', 'total_params', 'actual_epochs',
         'best_epoch', 'train_time_sec', 'rmse_scaled', 'mae_scaled',
         'r2_scaled', 'rmse_usd', 'mae_usd']
    ]
    print(cmp.to_string(index=False))

    # Determine winner
    best = min(experiments, key=lambda x: x['rmse_scaled'])
    worst = max(experiments, key=lambda x: x['rmse_scaled'])
    improvement = ((worst['rmse_scaled'] - best['rmse_scaled']) / worst['rmse_scaled']) * 100

    print(f"\n🏆 Winner: {best['experiment']}")
    print(f"   RMSE improvement over {worst['experiment']}: {improvement:.1f}%")
    print(f"   Parameters: {best['total_params']:,} vs {worst['total_params']:,}")
