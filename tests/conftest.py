import pytest
import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def synthetic_price_data():
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range('2022-01-01', periods=n, freq='D')
    returns = np.random.normal(0.001, 0.02, n)
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, n)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.02, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.02, n))),
        'price': prices,
        'volume': np.random.uniform(1e6, 1e7, n)
    })

@pytest.fixture
def synthetic_sequences():
    """Generate synthetic 3D sequence data."""
    np.random.seed(42)
    n_samples, seq_len, n_features = 100, 30, 5
    X = np.random.rand(n_samples, seq_len, n_features)
    y = np.random.rand(n_samples, 1)
    return X, y

@pytest.fixture
def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
