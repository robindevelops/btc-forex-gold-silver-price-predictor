import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_lstm_model(seq_len, n_features, learning_rate=0.001,
                     lstm_units=100, dense_units=50, dropout_rate=0.1):
    """
    Build and compile an LSTM model for price prediction.
    
    Args:
        seq_len (int): Length of the input sequence (time steps).
        n_features (int): Number of features in each time step.
        learning_rate (float): Learning rate for Adam optimizer.
        lstm_units (int): Number of units in each LSTM layer.
        dense_units (int): Number of units in the intermediate Dense layer.
        dropout_rate (float): Dropout rate after each LSTM layer.
        
    Returns:
        tf.keras.models.Sequential: Compiled LSTM model.
    """
    model = Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_features)),
        LSTM(lstm_units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(dense_units, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    
    return model

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from config import BEST_LSTM_CONFIG
    
    # Verify the model architecture
    seq_len = BEST_LSTM_CONFIG['seq_len']
    n_features = 5
    model = build_lstm_model(seq_len, n_features,
                             lstm_units=BEST_LSTM_CONFIG['lstm_units'],
                             dense_units=BEST_LSTM_CONFIG['dense_units'],
                             dropout_rate=BEST_LSTM_CONFIG['dropout_rate'])
    model.summary()

