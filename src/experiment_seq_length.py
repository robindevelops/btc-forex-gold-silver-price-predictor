"""
LSTM Sequence Length Experiment — 30 vs 60 vs 90 days.

Rebuilds sequences at each lookback length from the scaled CSVs,
trains an LSTM-100u model for each, evaluates on test set,
and appends results to results/experiment_log.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import MODELS_DIR, PROCESSED_DATA_DIR
from src.model_lstm import build_lstm_model
from src.preprocessing import create_sequences

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def inverse_transform_price(scaled_values, scaler, price_col_idx=0, n_features=16):
    """Inverse-transform scaled price column back to real USD."""
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, price_col_idx] = np.array(scaled_values).ravel()
    inv = scaler.inverse_transform(dummy)
    return inv[:, price_col_idx]


def build_sequences_for_btc(seq_len):
    """
    Build train+val and test sequences at a given seq_len
    from the saved scaled CSVs.

    Returns:
        X_train_full, y_train_full, X_test, y_test
    """
    train_df = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, 'btc_train_scaled.csv'),
        index_col='timestamp', parse_dates=True
    )
    val_df = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, 'btc_val_scaled.csv'),
        index_col='timestamp', parse_dates=True
    )
    test_df = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, 'btc_test_scaled.csv'),
        index_col='timestamp', parse_dates=True
    )

    X_train, y_train = create_sequences(train_df, seq_len=seq_len)
    X_val, y_val = create_sequences(val_df, seq_len=seq_len)
    X_test, y_test = create_sequences(test_df, seq_len=seq_len)

    # Merge train + val (same strategy as previous experiments)
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    print(f"  seq_len={seq_len}: train={X_train_full.shape[0]}, test={X_test.shape[0]}, features={X_train_full.shape[2]}")
    return X_train_full, y_train_full, X_test, y_test


def run_seq_experiment(seq_len, X_train, y_train, X_test, y_test, scaler,
                       lstm_units=100, dense_units=50,
                       learning_rate=0.001, batch_size=16,
                       epochs=150, patience=20):
    """Train LSTM with a given sequence length and return metrics."""
    n_features = X_train.shape[2]

    print(f"\n{'='*60}")
    print(f"  Experiment: seq_len={seq_len}, LSTM={lstm_units}u")
    print(f"  train={X_train.shape[0]}, test={X_test.shape[0]}")
    print(f"{'='*60}")

    model = build_lstm_model(
        seq_len, n_features,
        learning_rate=learning_rate,
        lstm_units=lstm_units,
        dense_units=dense_units
    )
    total_params = model.count_params()
    print(f"  Parameters: {total_params:,}")

    ckpt_path = os.path.join(MODELS_DIR, f'btc_lstm_seq{seq_len}_best.keras')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=ckpt_path, monitor='val_loss',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=7, min_lr=1e-6, verbose=1),
    ]

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

    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Scaled metrics
    rmse_s = np.sqrt(mean_squared_error(y_test, y_pred_scaled))
    mae_s = mean_absolute_error(y_test, y_pred_scaled)
    r2_s = r2_score(y_test, y_pred_scaled)

    # USD metrics
    y_test_real = inverse_transform_price(y_test, scaler, 0, n_features)
    y_pred_real = inverse_transform_price(y_pred_scaled, scaler, 0, n_features)
    rmse_usd = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae_usd = mean_absolute_error(y_test_real, y_pred_real)
    r2_usd = r2_score(y_test_real, y_pred_real)
    best_val_loss = min(history.history['val_loss'])

    print(f"\n  --- Results ---")
    print(f"  Epochs: {actual_epochs} (best @ {best_epoch}), Time: {train_time:.1f}s")
    print(f"  Scaled:  RMSE={rmse_s:.6f}  MAE={mae_s:.6f}  R²={r2_s:.6f}")
    print(f"  USD:     RMSE=${rmse_usd:,.2f}  MAE=${mae_usd:,.2f}")

    return {
        'timestamp': datetime.now().isoformat(),
        'experiment': f'seq{seq_len}-LSTM{lstm_units}u',
        'seq_len': seq_len,
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
    # Load scaler
    scaler = joblib.load(os.path.join(MODELS_DIR, 'btc_scaler.pkl'))

    # Fixed architecture: LSTM-100u (winner from units experiment)
    LSTM_UNITS = 100
    DENSE_UNITS = 50

    seq_lengths = [30, 60, 90]
    experiments = []

    for sl in seq_lengths:
        print(f"\n{'#'*60}")
        print(f"  Building sequences with seq_len={sl}")
        print(f"{'#'*60}")

        X_tr, y_tr, X_te, y_te = build_sequences_for_btc(sl)

        result = run_seq_experiment(
            seq_len=sl,
            X_train=X_tr, y_train=y_tr,
            X_test=X_te, y_test=y_te,
            scaler=scaler,
            lstm_units=LSTM_UNITS,
            dense_units=DENSE_UNITS,
        )
        experiments.append(result)

    # --- Append to experiment log ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, 'experiment_log.csv')

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        df = pd.concat([existing, pd.DataFrame(experiments)], ignore_index=True)
    else:
        df = pd.DataFrame(experiments)

    df.to_csv(log_path, index=False)
    print(f"\n📝 Experiment log updated: {log_path}")

    # --- Comparison table ---
    print(f"\n{'='*75}")
    print("EXPERIMENT COMPARISON — Sequence Length (30 vs 60 vs 90)")
    print(f"{'='*75}")
    cmp = pd.DataFrame(experiments)[
        ['experiment', 'seq_len', 'train_samples', 'test_samples',
         'actual_epochs', 'best_epoch', 'train_time_sec',
         'rmse_scaled', 'mae_scaled', 'r2_scaled', 'rmse_usd', 'mae_usd']
    ]
    print(cmp.to_string(index=False))

    # Winner
    best = min(experiments, key=lambda x: x['rmse_scaled'])
    print(f"\n🏆 Winner: {best['experiment']}  (RMSE={best['rmse_scaled']:.6f}, R²={best['r2_scaled']:.4f})")

    # Compare all three
    for exp in sorted(experiments, key=lambda x: x['rmse_scaled']):
        vs_best = ((exp['rmse_scaled'] - best['rmse_scaled']) / best['rmse_scaled']) * 100
        marker = " ← best" if exp == best else f" (+{vs_best:.1f}%)"
        print(f"  {exp['experiment']:>20s}:  RMSE={exp['rmse_scaled']:.6f}  R²={exp['r2_scaled']:.4f}{marker}")
