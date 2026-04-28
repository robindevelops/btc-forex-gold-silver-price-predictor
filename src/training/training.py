import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import MODELS_DIR
from src.models.model_lstm import build_lstm_model
from src.data.preprocessing import load_dataset

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def train_model(model, X_train, y_train, X_val, y_val,
                epochs=100, batch_size=32, patience=10, checkpoint_path=None):
    """
    Train an LSTM model with EarlyStopping and ModelCheckpoint callbacks.

    Args:
        model: Compiled Keras model.
        X_train: Training input sequences, shape (samples, seq_len, features).
        y_train: Training targets, shape (samples, 1).
        X_val: Validation input sequences.
        y_val: Validation targets.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Mini-batch size.
        checkpoint_path (str): File path to save the best model weights.
            Defaults to data/models/btc_lstm_best.h5.

    Returns:
        keras.callbacks.History: Training history object.
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(MODELS_DIR, 'btc_lstm_best.keras')

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # --- Callbacks ---
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    callbacks = [early_stopping, model_checkpoint, reduce_lr]

    # --- Train ---
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\nTraining complete.")
    print(f"  Best val_loss: {min(history.history['val_loss']):.6f}")
    print(f"  Best weights saved to: {checkpoint_path}")

    return history


def plot_loss_curve(history, save_path=None):
    """
    Plot training vs validation loss curve.

    Args:
        history: Keras History object from model.fit().
        save_path (str): File path to save the plot. Defaults to results/btc_lstm_loss_curve.png.
    """
    if save_path is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_path = os.path.join(RESULTS_DIR, 'btc_lstm_loss_curve.png')

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-o', markersize=3, label='Training Loss', linewidth=1.5)
    plt.plot(epochs, val_loss, 'r-o', markersize=3, label='Validation Loss', linewidth=1.5)

    # Mark the best epoch
    best_epoch = np.argmin(val_loss) + 1
    best_val = min(val_loss)
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7,
                label=f'Best Epoch ({best_epoch}, val_loss={best_val:.6f})')

    plt.title('BTC LSTM — Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to: {save_path}")


if __name__ == "__main__":
    # --- Tuned hyperparameters (v2) ---
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 16
    EPOCHS = 200
    PATIENCE = 15

    # --- Load preprocessed BTC sequences ---
    print("Loading BTC dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset('Bitcoin')

    # Combine train + val for more training data, use test as validation
    # This is acceptable for time-series when val set is very small
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    print(f"  X_train (original):  {X_train.shape}")
    print(f"  X_val   (original):  {X_val.shape}")
    print(f"  X_train (combined):  {X_train_full.shape}")
    print(f"  X_test  (held-out):  {X_test.shape}")
    print(f"  Hyperparams: lr={LEARNING_RATE}, batch={BATCH_SIZE}, epochs={EPOCHS}, patience={PATIENCE}")

    # --- Build model ---
    seq_len = X_train_full.shape[1]
    n_features = X_train_full.shape[2]
    model = build_lstm_model(seq_len, n_features, learning_rate=LEARNING_RATE)
    model.summary()

    # --- Train (use test set as validation for monitoring only) ---
    history = train_model(
        model, X_train_full, y_train_full, X_test, y_test,
        epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE
    )

    # --- Save training history ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(RESULTS_DIR, 'btc_lstm_history.npy'), history.history)
    print(f"Training history saved to: {os.path.join(RESULTS_DIR, 'btc_lstm_history.npy')}")

    # --- Plot loss curve ---
    plot_loss_curve(history)
