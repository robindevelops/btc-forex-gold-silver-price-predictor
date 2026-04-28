import unittest
import pandas as pd
import numpy as np
import os
import sys

# Setup path so tests can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessing import DataCleaner, create_sequences
from config import MODELS_DIR, BEST_LSTM_CONFIG
from tensorflow.keras.models import load_model

class TestCryptoPipeline(unittest.TestCase):
    
    def setUp(self):
        """Creates a dummy raw dataset before each test."""
        # Generate 250 days of dummy data to ensure we pass the 100 row validation after warmups
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')
        prices = np.linspace(100, 200, 250) + np.random.normal(0, 5, 250)
        volumes = np.random.randint(1000, 5000, 250)
        
        self.dummy_df = pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': volumes
        })
        # Inject a NaN to test the forward-fill missing data handler
        self.dummy_df.loc[50, 'price'] = np.nan
        
    def test_data_cleaning_and_features(self):
        """Tests if DataCleaner handles NaNs and calculates technical indicators."""
        cleaner = DataCleaner('Bitcoin')
        cleaner.df = self.dummy_df.copy()
        
        # We only want to test the cleaning logic, not the file saving/loading
        cleaner.clean_data()
        
        # 1. Check NaN handling
        self.assertFalse(cleaner.df['price'].isna().any(), "DataCleaner failed to handle missing values.")
        
        # 2. Check Feature Engineering
        expected_columns = ['price', 'volume', 'SMA_30', 'EMA_30', 'RSI', 'MACD', 'BB_Upper']
        for col in expected_columns:
            self.assertIn(col, cleaner.df.columns, f"Technical indicator {col} was not generated.")
            
        # 3. Check Warmup Period Dropping
        # Because SMA_60 requires 60 days of data, the final dataframe should be shorter
        self.assertTrue(len(cleaner.df) < 250, "DataCleaner failed to drop NaN warmup periods.")
        
    def test_sequence_creation(self):
        """Tests if the sequence generator outputs the correct 3D tensor shape for LSTMs."""
        # Create a dummy scaled dataset: 200 rows, 5 features
        df = pd.DataFrame(np.random.rand(200, 5), columns=['price', 'feat_2', 'feat_3', 'feat_4', 'feat_5'])
        seq_len = 30
        
        X, y = create_sequences(df, seq_len=seq_len, target_col='price')
        
        # Expected samples = total_rows - sequence_length
        expected_samples = 200 - seq_len
        
        self.assertEqual(X.shape, (expected_samples, seq_len, 5), "Sequence X shape (3D tensor) is incorrect.")
        self.assertEqual(y.shape, (expected_samples, 1), "Sequence y target shape is incorrect.")
        
    def test_model_loading_and_prediction(self):
        """Tests if the saved Keras model can be loaded and successfully run inference."""
        model_path = os.path.join(MODELS_DIR, 'btc_lstm_final.keras')
        
        if not os.path.exists(model_path):
            self.skipTest("Final BTC model not found. Skipping prediction test.")
            
        # 1. Test Model Loading
        model = load_model(model_path)
        self.assertIsNotNone(model, "Failed to load the LSTM Keras model.")
        
        # 2. Test Prediction Output
        seq_len = BEST_LSTM_CONFIG['seq_len']
        n_features = 16 # Based on our standard pipeline
        
        # Create a dummy input sequence (1 sample, 30 timesteps, 16 features)
        dummy_input = np.random.rand(1, seq_len, n_features)
        
        prediction = model.predict(dummy_input, verbose=0)
        
        # Prediction should be a single continuous value
        self.assertEqual(prediction.shape, (1, 1), "Model prediction output shape is incorrect.")
        self.assertFalse(np.isnan(prediction[0][0]), "Model predicted NaN.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
