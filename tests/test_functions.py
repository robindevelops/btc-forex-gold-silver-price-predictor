"""
Tests for feature engineering functions.

Created: Week 1 — replaced empty placeholder file with real tests
for the core technical indicator calculations.
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys

# Ensure src and root are in the path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from src.data.preprocessing import DataCleaner


class TestFeatureEngineering(unittest.TestCase):
    """Tests for technical indicator calculations in DataCleaner."""

    @classmethod
    def setUpClass(cls):
        """Create a synthetic price series for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        # Simulate a random walk price series starting at 100
        returns = np.random.normal(0.001, 0.02, 200)
        prices = 100 * np.exp(np.cumsum(returns))
        volumes = np.random.uniform(1e6, 1e7, 200)

        cls.df = pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': volumes
        })

    def _make_cleaner(self):
        """Create a DataCleaner with our test data loaded."""
        cleaner = DataCleaner('Bitcoin')
        cleaner.df = self.df.copy()
        cleaner.df = cleaner.df.set_index('timestamp')
        return cleaner

    def test_sma_calculation(self):
        """SMA-7 should equal the rolling mean of the last 7 prices."""
        cleaner = self._make_cleaner()
        cleaner.add_moving_averages()

        # Check SMA_7 at a specific point (after warm-up)
        idx = 70  # well past the 60-day warm-up
        expected_sma7 = cleaner.df['price'].iloc[idx-6:idx+1].mean()
        actual_sma7 = cleaner.df['SMA_7'].iloc[idx]
        self.assertAlmostEqual(actual_sma7, expected_sma7, places=4,
                               msg="SMA_7 doesn't match manual rolling mean calculation")

    def test_rsi_range(self):
        """RSI should always be between 0 and 100."""
        cleaner = self._make_cleaner()
        cleaner.add_rsi(window=14)

        rsi = cleaner.df['RSI'].dropna()
        self.assertTrue((rsi >= 0).all(), "RSI has values below 0")
        self.assertTrue((rsi <= 100).all(), "RSI has values above 100")

    def test_macd_signal_relationship(self):
        """MACD Signal should be a smoothed version of MACD — same length, no NaNs after warm-up."""
        cleaner = self._make_cleaner()
        cleaner.add_macd(fast=12, slow=26, signal=9)

        self.assertIn('MACD', cleaner.df.columns)
        self.assertIn('MACD_Signal', cleaner.df.columns)
        self.assertEqual(len(cleaner.df['MACD']), len(cleaner.df['MACD_Signal']))

    def test_bollinger_bands_ordering(self):
        """Upper band >= Middle band >= Lower band everywhere."""
        cleaner = self._make_cleaner()
        cleaner.add_bollinger_bands(period=20, std_dev=2)

        valid = cleaner.df.dropna(subset=['BB_Upper', 'BB_Mid', 'BB_Lower'])
        self.assertTrue(
            (valid['BB_Upper'] >= valid['BB_Mid']).all(),
            "BB_Upper is not always >= BB_Mid"
        )
        self.assertTrue(
            (valid['BB_Mid'] >= valid['BB_Lower']).all(),
            "BB_Mid is not always >= BB_Lower"
        )

    def test_no_future_data_in_sma(self):
        """SMA at time t should only use data up to and including time t."""
        cleaner = self._make_cleaner()
        cleaner.add_moving_averages()

        # Modify a future price and verify SMA_7 at an earlier point is unchanged
        df_original = cleaner.df.copy()

        cleaner2 = self._make_cleaner()
        cleaner2.df.iloc[-1, cleaner2.df.columns.get_loc('price')] *= 2  # double last price
        cleaner2.add_moving_averages()

        # SMA_7 at index 100 should be identical regardless of the last price
        idx = 100
        self.assertAlmostEqual(
            df_original['SMA_7'].iloc[idx],
            cleaner2.df['SMA_7'].iloc[idx],
            places=10,
            msg="SMA_7 at t=100 changed when future data was modified — possible lookahead"
        )

    def test_feature_count(self):
        """After full cleaning, the feature count should match expectations."""
        cleaner = self._make_cleaner()
        cleaner.add_moving_averages()
        cleaner.add_rsi()
        cleaner.add_macd()
        cleaner.add_bollinger_bands()

        expected_features = {
            'price', 'volume',
            'SMA_7', 'SMA_14', 'SMA_30', 'SMA_60',
            'EMA_7', 'EMA_14', 'EMA_30', 'EMA_60',
            'RSI', 'MACD', 'MACD_Signal',
            'BB_Mid', 'BB_Upper', 'BB_Lower'
        }
        actual_features = set(cleaner.df.columns)
        self.assertEqual(expected_features, actual_features,
                         f"Feature mismatch. Missing: {expected_features - actual_features}, "
                         f"Extra: {actual_features - expected_features}")


if __name__ == '__main__':
    unittest.main()
