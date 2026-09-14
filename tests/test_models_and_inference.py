"""Model registry contract and end-to-end inference on the saved artefacts (if present)."""
import os
import numpy as np
import pandas as pd
import pytest
from config import MODELS_DIR, ASSETS
from src.models.registry import make_model


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
def test_served_prediction_is_sane_and_combines_every_trained_model(asset):
    from src.inference.prediction import predict_next_day, combined_members, COMBINED
    r = predict_next_day(asset)
    assert r['model_used'] == COMBINED and r['is_served_model']
    assert r['current_price'] > 0 and r['predicted_price'] > 0
    assert abs(r['predicted_return_pct']) < 20              # a next-day forecast of >20% would be a bug
    assert np.isclose(r['predicted_price'], r['current_price'] * np.exp(r['predicted_return_pct'] / 100))
    assert r['direction'] in ('UP', 'DOWN') and r['horizon_days'] == 1 and 'disclaimer' in r
    members = r['individual']
    assert [m['model'] for m in members] == combined_members(asset) and len(members) == 6
    assert np.isclose(r['predicted_return_pct'], np.mean([m['predicted_return_pct'] for m in members]))   # equal-weight mean
    for m in members:
        assert np.isfinite(m['predicted_price']) and abs(m['predicted_return_pct']) < 20


@pytest.mark.skipif(not os.path.exists(os.path.join(MODELS_DIR, 'model_status.json')), reason="models not trained")
def test_combined_test_row_is_the_mean_of_its_members():
    """The evaluation script's 'Combined' row must be exactly the equal-weight mean of the members' stored test predictions."""
    import pandas as pd
    from config import RESULTS_DIR, load_model_status
    st = load_model_status()['Gold']
    h = pd.read_csv(os.path.join(RESULTS_DIR, 'predictions', 'gold_test_predictions.csv'))
    if 'pred_return_Combined' not in h.columns:
        pytest.skip("evaluation not re-run with the Combined forecast")
    mean = np.mean([h[f'pred_return_{m}'] for m in st['combined_members']], axis=0)
    assert np.allclose(h['pred_return_Combined'], mean)


@pytest.mark.skipif(not os.path.exists(os.path.join(MODELS_DIR, 'model_status.json')), reason="models not trained")
def test_predict_for_date_uses_only_past_data_and_reveals_actual(monkeypatch):
    import src.inference.prediction as pr
    from config import VAL_END
    f = pr.load_features('Gold', live=False)                    # the frozen dataset the demo uses
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
    monkeypatch.setattr(pr, 'load_features', lambda asset, live=True: g)
    r2 = pr.predict_for_date('Gold', str(day.date()))
    assert np.isclose(r['predicted_price'], r2['predicted_price'])
    assert not np.isclose(r['actual_price'], r2['actual_price'])   # the revealed actual DID change → the perturbation was applied


# ────────────────────────────────────────── failure modes must be explicit, never silent
@pytest.mark.skipif(not os.path.exists(os.path.join(MODELS_DIR, 'model_status.json')), reason="models not trained")
def test_corrupt_live_data_raises_instead_of_returning_nan(monkeypatch):
    import src.inference.prediction as pr
    g = pr.load_features('Gold', live=False).copy()
    g.iloc[-1, g.columns.get_loc('RSI')] = np.nan                      # a NaN in the newest row
    monkeypatch.setattr(pr, 'load_features', lambda asset, live=True: g)
    with pytest.raises(ValueError):
        pr.predict_next_day('Gold')


def test_missing_evaluation_is_reported_clearly(monkeypatch):
    import src.inference.prediction as pr
    monkeypatch.setattr(pr, 'load_model_status', lambda: {})           # no model_status.json → no target statistics
    with pytest.raises(FileNotFoundError):
        pr._target_stats('Gold')


def test_live_sync_survives_network_failure(monkeypatch):
    import src.data.sync_live_data as sl
    monkeypatch.setattr(sl, 'fetch_all_external_data', lambda out_dir: (_ for _ in ()).throw(ConnectionError('offline')))
    assert sl.update_live_data('Bitcoin', refresh_external=True) is False   # returns False, never raises
