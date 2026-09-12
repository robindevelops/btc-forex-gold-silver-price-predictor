"""
LSTM model for next-day log-return prediction (same shape as the GRU for a fair comparison).
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_lstm_model(seq_len, n_features, units=32, dropout=0.2, learning_rate=0.001):
    model = Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_features)),
        LSTM(units),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model
