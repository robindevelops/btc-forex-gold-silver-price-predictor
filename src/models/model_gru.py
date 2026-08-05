"""
GRU Model Architecture.

Week 5: Lighter sequential alternative to LSTM.
GRU (Gated Recurrent Unit) has fewer parameters than LSTM, making it less prone
to overfitting on small datasets while still capturing sequential dependencies.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_gru_model(seq_len, n_features, learning_rate=0.001,
                    gru_units=100, dense_units=50, dropout_rate=0.1):
    """
    Build and compile a GRU model for price prediction.
    
    Args:
        seq_len (int): Length of the input sequence (time steps).
        n_features (int): Number of features in each time step.
        learning_rate (float): Learning rate for Adam optimizer.
        gru_units (int): Number of units in each GRU layer.
        dense_units (int): Number of units in the intermediate Dense layer.
        dropout_rate (float): Dropout rate after each GRU layer.
        
    Returns:
        tf.keras.models.Sequential: Compiled GRU model.
    """
    model = Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_features)),
        GRU(gru_units, return_sequences=True),
        Dropout(dropout_rate),
        GRU(gru_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(dense_units, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    
    return model
