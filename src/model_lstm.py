import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_lstm_model(seq_len, n_features, learning_rate=0.001,
                     lstm_units=50, dense_units=25, dropout_rate=0.2):
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
        LSTM(lstm_units, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(dense_units),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    
    return model

if __name__ == "__main__":
    # Verify the model architecture
    seq_len = 60
    n_features = 5
    model = build_lstm_model(seq_len, n_features)
    model.summary()
