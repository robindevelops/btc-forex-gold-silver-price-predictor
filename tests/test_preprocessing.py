import unittest
import numpy as np
import os
import pandas as pd
import sys

# Ensure src and root are in the path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from src.preprocessing import load_dataset, create_sequences
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

class TestPreprocessing(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # We test specifically on Bitcoin as it represents the overall pipeline format
        cls.asset_name = 'Bitcoin'
        cls.prefix = 'btc'
        
        # Load datasets to verify correct split sizes & scaler bounds
        cls.train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f"{cls.prefix}_train_scaled.csv"), index_col='timestamp')
        cls.val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f"{cls.prefix}_val_scaled.csv"), index_col='timestamp')
        cls.test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f"{cls.prefix}_test_scaled.csv"), index_col='timestamp')
        cls.features_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f"{cls.prefix}_features.csv"), index_col='timestamp')
        
        # Load sequences to verify shapes
        cls.X_train, cls.y_train, cls.X_val, cls.y_val, cls.X_test, cls.y_test = load_dataset(cls.asset_name)

    def test_split_sizes(self):
        """Test if Train + Val + Test roughly equals total features length"""
        total_split = len(self.train_df) + len(self.val_df) + len(self.test_df)
        total_original = len(self.features_df)
        self.assertEqual(total_split, total_original, 
                         f"Split size mismatch: {total_split} total splits vs {total_original} original rows")
        
        # Verify ~70% / 15% / 15% proportions
        self.assertAlmostEqual(len(self.train_df) / total_original, 0.70, places=1)
        self.assertAlmostEqual(len(self.val_df) / total_original, 0.15, places=1)
        self.assertAlmostEqual(len(self.test_df) / total_original, 0.15, places=1)

    def test_scaler_range(self):
        """Test if the scaled data is within 0 to 1 range (MinMaxScaler verification)"""
        # We test the training data bounds: Min should be >= 0, Max should be <= 1
        # Epsilon buffer added for floating point errors common in ML processing
        eps = 1e-5
        
        train_min = self.train_df.min().min()
        train_max = self.train_df.max().max()
        
        self.assertTrue(train_min >= 0.0 - eps, f"Train min < 0: {train_min}")
        self.assertTrue(train_max <= 1.0 + eps, f"Train max > 1: {train_max}")

    def test_no_nan_values(self):
        """Test that dataframes and sequences contain zero NaNs"""
        self.assertEqual(self.features_df.isna().sum().sum(), 0, "Original features contain NaNs")
        self.assertEqual(self.train_df.isna().sum().sum(), 0, "Train scaled contains NaNs")
        
        self.assertFalse(np.isnan(self.X_train).any(), "X_train sequence contains NaNs")
        self.assertFalse(np.isnan(self.y_train).any(), "y_train sequence contains NaNs")

    def test_sequence_shapes(self):
        """Test that the 3D X sequences and 2D y targets have corresponding and correct shapes"""
        seq_len = 60
        features_num = self.features_df.shape[1]
        
        # 1. Check sequence length
        self.assertEqual(self.X_train.shape[1], seq_len, "X_train Sequence length is not 60")
        
        # 2. Check feature dimension
        self.assertEqual(self.X_train.shape[2], features_num, f"X_train feature count is not {features_num}")
        
        # 3. Check X and y sample alignment
        self.assertEqual(self.X_train.shape[0], self.y_train.shape[0], "X_train and y_train samples misaligned")
        self.assertEqual(self.X_val.shape[0], self.y_val.shape[0], "X_val and y_val samples misaligned")
        self.assertEqual(self.X_test.shape[0], self.y_test.shape[0], "X_test and y_test samples misaligned")

if __name__ == '__main__':
    unittest.main()
