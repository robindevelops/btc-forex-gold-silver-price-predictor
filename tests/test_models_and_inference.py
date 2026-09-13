"""Model registry contract and end-to-end inference on the saved artefacts (if present)."""
import os
import numpy as np
import pandas as pd
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
    m = make_model('Naive', constant=0.42)
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


@pytest.mark.skipif(not os.path.exists(os.path.join(MODELS_DIR, 'model_status.json')), reason="models not trained")
def test_predict_for_date_uses_only_past_data_and_reveals_actual(monkeypatch):
    import src.inference.prediction as pr
    from config import VAL_END
    f = pr.load_features('Gold')
    day = f.loc[pd.Timestamp(VAL_END):].index[5]                 # a day inside the unseen test period
    r = pr.predict_for_date('Gold', str(day.date()))
    assert r['in_unseen_test_period'] and r['actual_price'] is not None
    pos = f.index.get_loc(day)
    assert np.isclose(r['current_price'], f['price'].iloc[pos]) and np.isclose(r['actual_price'], f['price'].iloc[pos + 1])
    assert np.isclose(r['error_pct'], (r['predicted_price'] - r['actual_price']) / r['actual_price'] * 100)
    # perturbing every FUTURE row (after `day`) must not change the prediction — genuine no-look-ahead check
    g = f.copy()
    num = g.columns[g.dtypes != object]
    g.loc[g.index > day, num] = g.loc[g.index > day, num] * 1.37 + 0.01
    monkeypatch.setattr(pr, 'load_features', lambda asset: g)
    r2 = pr.predict_for_date('Gold', str(day.date()))
    assert np.isclose(r['predicted_price'], r2['predicted_price'])
    assert not np.isclose(r['actual_price'], r2['actual_price'])   # the revealed actual DID change → the perturbation was applied
