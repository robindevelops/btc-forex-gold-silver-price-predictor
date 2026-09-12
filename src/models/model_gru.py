"""
GRU model for next-day log-return prediction.

A single GRU layer with dropout. The number of units is a tuned hyper-parameter
(32 or 64 in the walk-forward search); with ~500–700 training windows a deeper
2×100-unit stack (~130k parameters) over-fitted badly in the original experiments.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_gru_model(seq_len, n_features, units=32, dropout=0.2, learning_rate=0.001):
    model = Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_features)),
        GRU(units),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model
