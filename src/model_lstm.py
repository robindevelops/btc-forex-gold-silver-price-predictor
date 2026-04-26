import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_lstm_model(seq_len, n_features, learning_rate=0.001):
    """
    Build and compile an LSTM model for price prediction.
    
    Args:
        seq_len (int): Length of the input sequence (time steps).
        n_features (int): Number of features in each time step.
        learning_rate (float): Learning rate for Adam optimizer.
        
    Returns:
        tf.keras.models.Sequential: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
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
