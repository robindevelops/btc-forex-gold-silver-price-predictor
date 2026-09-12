"""Model registry contract and end-to-end inference on the saved artefacts (if present)."""
import os
import numpy as np
import pytest
from config import MODELS_DIR, ASSETS, get_prefix
from src.models.registry import make_model, TABULAR


@pytest.mark.parametrize('name', ['Ridge', 'RandomForest', 'LightGBM', 'CatBoost'])
def test_tabular_models_fit_predict_and_two_phase(name, synthetic_sequences):
    X, y = synthetic_sequences
    Xt = X[:, -1, :]
    m = make_model(name, {'n_estimators': 20} if name in ('RandomForest', 'LightGBM') else ({'iterations': 20} if name == 'CatBoost' else None))
    m.fit(X[:70], Xt[:70], y[:70], X[70:], Xt[70:], y[70:], final=True)
    assert m.predict(X[70:], Xt[70:]).shape == (30,)
    assert isinstance(m.fit_info, dict)


def test_naive_zero_returns_constant(synthetic_sequences):
    X, y = synthetic_sequences
    m = make_model('Naive-Zero', scaled_zero=0.42)
    assert np.allclose(m.predict(X, X[:, -1, :]), 0.42)


@pytest.mark.skipif(not os.path.exists(os.path.join(MODELS_DIR, 'model_status.json')), reason="models not trained")
@pytest.mark.parametrize('asset', ASSETS)
def test_served_prediction_is_sane(asset):
    from src.inference.prediction import predict_next_day
    r = predict_next_day(asset)
    assert r['current_price'] > 0 and r['predicted_price'] > 0
    assert abs(r['predicted_return_pct']) < 20              # a next-day forecast of >20% would be a bug
    assert np.isclose(r['predicted_price'], r['current_price'] * np.exp(r['predicted_return_pct'] / 100))
    assert r['direction'] in ('UP', 'DOWN') and r['horizon_days'] == 1 and 'disclaimer' in r
