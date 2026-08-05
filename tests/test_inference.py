import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.utils.inverse_transform import reconstruct_price
from src.data.preprocessing import create_sequences

def test_create_sequences():
    data = pd.DataFrame({
        'price': np.arange(100),
        'log_return': np.random.randn(100)
    })
    
    seq_len = 10
    X, y = create_sequences(data, seq_len=seq_len, target_col='log_return')
    
    assert X.shape == (90, 10, 2)
    assert y.shape == (90, 1)
    
    # Check if target is correct (log_return at i+seq_len)
    assert np.allclose(y.ravel(), data['log_return'].values[10:])

def test_reconstruct_price_direct():
    scaler = MagicMock()
    scaler.n_features_in_ = 5
    
    def inverse_transform(X):
        # Dummy inverse transform that just multiplies by 100 for test
        return X * 100
        
    scaler.inverse_transform.side_effect = inverse_transform
    
    y_pred_scaled = np.array([[0.5]])
    X_seq_scaled = np.zeros((1, 10, 5))
    
    price = reconstruct_price(
        y_pred_scaled, X_seq_scaled, scaler, 
        target_col_idx=0, price_col_idx=0
    )
    
    assert price.shape == (1,)
    assert price[0] == 50.0  # 0.5 * 100

def test_reconstruct_price_log_return():
    scaler = MagicMock()
    scaler.n_features_in_ = 3
    
    def inverse_transform(X):
        # return as is, assuming scaled=unscaled for this test, except scale up price a bit
        res = X.copy()
        res[:, 0] = res[:, 0] * 1000  # price col
        return res
        
    scaler.inverse_transform.side_effect = inverse_transform
    
    # log_return = 0.01
    y_pred_scaled = np.array([[0.01]])
    
    # t-1 price = 0.5 (scaled) -> 500 (unscaled)
    X_seq_scaled = np.zeros((1, 10, 3))
    X_seq_scaled[0, -1, 0] = 0.5
    
    price = reconstruct_price(
        y_pred_scaled, X_seq_scaled, scaler, 
        target_col_idx=2, price_col_idx=0
    )
    
    assert price.shape == (1,)
    # price = 500 * exp(0.01)
    assert np.isclose(price[0], 500 * np.exp(0.01))

@patch("src.inference.prediction.load_inference_data")
def test_missing_model_file_handling(mock_load):
    from src.inference.prediction import predict_next_day_safe
    
    # Mock data to avoid FileNotFoundError on missing data
    mock_load.return_value = (np.zeros((1, 30, 5)), MagicMock(), 0, pd.DataFrame({'price': [100.0]}))
    
    with patch("src.inference.prediction.MODEL_STATUS", {"Bitcoin": {"best_model": "NonExistentModel"}}):
        # Should return None and not crash when inference is not implemented
        res = predict_next_day_safe("Bitcoin")
        assert res is None
