import pytest
import numpy as np
import os
import tempfile

try:
    import tensorflow as tf
    HAS_TF = True
except (ImportError, OSError, PermissionError):
    HAS_TF = False

from src.models.model_lgbm import build_lgbm_model, save_lgbm_model, load_lgbm_model

# Conditionally import Keras model builders
if HAS_TF:
    from src.models.model_lstm import build_lstm_model
    from src.models.model_gru import build_gru_model

@pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")
def test_lstm_model_build(synthetic_sequences):
    X, y = synthetic_sequences
    seq_len = X.shape[1]
    n_features = X.shape[2]
    
    model = build_lstm_model(seq_len=seq_len, n_features=n_features)
    assert isinstance(model, tf.keras.models.Sequential)
    
    # Test output shape
    output = model.predict(X[:2], verbose=0)
    assert output.shape == (2, 1)
    
    # Test compilation
    assert model.optimizer is not None
    assert not np.isnan(output).any()

@pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")
def test_gru_model_build(synthetic_sequences):
    X, y = synthetic_sequences
    seq_len = X.shape[1]
    n_features = X.shape[2]
    
    model = build_gru_model(seq_len=seq_len, n_features=n_features)
    assert isinstance(model, tf.keras.models.Sequential)
    
    # Test output shape
    output = model.predict(X[:2], verbose=0)
    assert output.shape == (2, 1)
    
    # Test compilation
    assert model.optimizer is not None
    assert not np.isnan(output).any()

def test_lgbm_model_build_and_fit(synthetic_sequences):
    X, y = synthetic_sequences
    # LightGBM requires 2D input (n_samples, n_features)
    X_2d = X.reshape(X.shape[0], -1)
    
    model = build_lgbm_model()
    model.fit(X_2d, y.ravel())
    
    predictions = model.predict(X_2d[:2])
    assert predictions.shape == (2,)
    assert not np.isnan(predictions).any()

def test_model_serialization_lgbm(synthetic_sequences):
    X, y = synthetic_sequences
    X_2d = X.reshape(X.shape[0], -1)
    
    model = build_lgbm_model()
    model.fit(X_2d, y.ravel())
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "model.joblib")
        save_lgbm_model(model, filepath)
        assert os.path.exists(filepath)
        
        loaded_model = load_lgbm_model(filepath)
        preds_original = model.predict(X_2d[:5])
        preds_loaded = loaded_model.predict(X_2d[:5])
        
        np.testing.assert_array_almost_equal(preds_original, preds_loaded)

@pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")
def test_model_hyperparameters():
    lstm = build_lstm_model(seq_len=10, n_features=5, lstm_units=64, dense_units=32, dropout_rate=0.3, learning_rate=0.01)
    assert lstm.layers[0].units == 64
    
    gru = build_gru_model(seq_len=10, n_features=5, gru_units=32, dense_units=16, dropout_rate=0.1, learning_rate=0.005)
    assert gru.layers[0].units == 32
    
    lgbm = build_lgbm_model(n_estimators=50, learning_rate=0.1, max_depth=4)
    assert lgbm.n_estimators == 50
    assert lgbm.learning_rate == 0.1
    assert lgbm.max_depth == 4
