"""
LSTM Dropout Experiment — 0.1 vs 0.2 vs 0.3.

Uses the best config from previous experiments (100 units, seq_len=30).
Trains three variants, evaluates each, and appends to experiment_log.csv.
Then picks the overall best combination across ALL logged experiments.
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
    return scaler.inverse_transform(dummy)[:, price_col_idx]


def build_sequences(seq_len=30):
    """Build train+val merged and test sequences at a given seq_len."""
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'btc_train_scaled.csv'),
                           index_col='timestamp', parse_dates=True)
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'btc_val_scaled.csv'),
                         index_col='timestamp', parse_dates=True)
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'btc_test_scaled.csv'),
                          index_col='timestamp', parse_dates=True)

    X_train, y_train = create_sequences(train_df, seq_len=seq_len)
    X_val, y_val = create_sequences(val_df, seq_len=seq_len)
    X_test, y_test = create_sequences(test_df, seq_len=seq_len)

    X_tr = np.concatenate([X_train, X_val], axis=0)
    y_tr = np.concatenate([y_train, y_val], axis=0)
    return X_tr, y_tr, X_test, y_test


def run_dropout_experiment(dropout_rate, X_train, y_train, X_test, y_test, scaler,
                           seq_len=30, lstm_units=100, dense_units=50,
                           learning_rate=0.001, batch_size=16,
                           epochs=150, patience=20):
    """Train a single dropout variant and return metrics."""
    n_features = X_train.shape[2]

    print(f"\n{'='*60}")
    print(f"  Dropout={dropout_rate}, LSTM={lstm_units}u, seq={seq_len}")
    print(f"{'='*60}")

    model = build_lstm_model(
        seq_len, n_features,
        learning_rate=learning_rate,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout_rate=dropout_rate
    )
    total_params = model.count_params()

    ckpt_path = os.path.join(MODELS_DIR, f'btc_lstm_drop{int(dropout_rate*100)}_best.keras')

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
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=0
    )
    train_time = (datetime.now() - start_time).total_seconds()
    actual_epochs = len(history.history['loss'])
    best_epoch = np.argmin(history.history['val_loss']) + 1

    y_pred_scaled = model.predict(X_test, verbose=0)
    rmse_s = np.sqrt(mean_squared_error(y_test, y_pred_scaled))
    mae_s = mean_absolute_error(y_test, y_pred_scaled)
    r2_s = r2_score(y_test, y_pred_scaled)

    y_test_real = inverse_transform_price(y_test, scaler, 0, n_features)
    y_pred_real = inverse_transform_price(y_pred_scaled, scaler, 0, n_features)
    rmse_usd = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae_usd = mean_absolute_error(y_test_real, y_pred_real)
    r2_usd = r2_score(y_test_real, y_pred_real)

    print(f"  {actual_epochs} epochs (best @ {best_epoch}), {train_time:.0f}s")
    print(f"  RMSE={rmse_s:.6f}  MAE={mae_s:.6f}  R²={r2_s:.6f}  (${rmse_usd:,.0f})")

    return {
        'timestamp': datetime.now().isoformat(),
        'experiment': f'drop{int(dropout_rate*100)}-seq{seq_len}-LSTM{lstm_units}u',
        'seq_len': seq_len,
        'lstm_units': lstm_units,
        'dense_units': dense_units,
        'dropout_rate': dropout_rate,
        'total_params': total_params,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'max_epochs': epochs,
        'actual_epochs': actual_epochs,
        'best_epoch': best_epoch,
        'best_val_loss': min(history.history['val_loss']),
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
    # --- Config: best settings from prior experiments ---
    SEQ_LEN = 30
    LSTM_UNITS = 100
    DENSE_UNITS = 50

    # --- Load data ---
    print("Building sequences (seq_len=30)...")
    X_tr, y_tr, X_te, y_te = build_sequences(SEQ_LEN)
    print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")

    scaler = joblib.load(os.path.join(MODELS_DIR, 'btc_scaler.pkl'))

    # --- Run dropout experiments ---
    experiments = []
    for dr in [0.1, 0.2, 0.3]:
        result = run_dropout_experiment(
            dropout_rate=dr,
            X_train=X_tr, y_train=y_tr,
            X_test=X_te, y_test=y_te,
            scaler=scaler,
            seq_len=SEQ_LEN,
            lstm_units=LSTM_UNITS,
            dense_units=DENSE_UNITS,
        )
        experiments.append(result)

    # --- Append to experiment log ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, 'experiment_log.csv')

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        df_all = pd.concat([existing, pd.DataFrame(experiments)], ignore_index=True)
    else:
        df_all = pd.DataFrame(experiments)

    df_all.to_csv(log_path, index=False)
    print(f"\n📝 Experiment log updated: {log_path}  ({len(df_all)} total runs)")

    # --- Dropout comparison ---
    print(f"\n{'='*70}")
    print(f"DROPOUT COMPARISON — 0.1 vs 0.2 vs 0.3  (seq={SEQ_LEN}, LSTM={LSTM_UNITS}u)")
    print(f"{'='*70}")
    cmp = pd.DataFrame(experiments)[
        ['experiment', 'dropout_rate', 'actual_epochs', 'best_epoch',
         'train_time_sec', 'rmse_scaled', 'mae_scaled', 'r2_scaled',
         'rmse_usd', 'mae_usd']
    ]
    print(cmp.to_string(index=False))

    best_drop = min(experiments, key=lambda x: x['rmse_scaled'])
    print(f"\n🏆 Best dropout: {best_drop['dropout_rate']}  "
          f"(RMSE={best_drop['rmse_scaled']:.6f}, R²={best_drop['r2_scaled']:.4f})")

    # --- Overall best across ALL experiments ---
    print(f"\n{'='*70}")
    print(f"OVERALL BEST CONFIGURATION (across all {len(df_all)} experiments)")
    print(f"{'='*70}")

    # Only compare experiments with R² > 0.8 (exclude unreliable small test sets)
    reliable = df_all[df_all['r2_scaled'] > 0.8].copy()
    if len(reliable) > 0:
        best_idx = reliable['rmse_scaled'].idxmin()
        best_row = reliable.loc[best_idx]
        print(f"\n  🏆 {best_row['experiment']}")
        print(f"     RMSE:    {best_row['rmse_scaled']:.6f}  (${best_row['rmse_usd']:,.0f})")
        print(f"     MAE:     {best_row['mae_scaled']:.6f}  (${best_row['mae_usd']:,.0f})")
        print(f"     R²:      {best_row['r2_scaled']:.6f}")
        seq = best_row.get('seq_len', 'N/A')
        units = best_row.get('lstm_units', 'N/A')
        drop = best_row.get('dropout_rate', 'N/A')
        print(f"     Config:  seq_len={seq}, lstm_units={units}, dropout={drop}")
    else:
        print("  No experiments with R² > 0.8 found.")
