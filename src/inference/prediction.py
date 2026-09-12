"""
Next-day prediction for the dashboard / API.

    from src.inference.prediction import predict_next_day
    predict_next_day('Gold')            # served model (data/models/model_status.json)
    predict_next_day('Gold', 'GRU')     # any trained model

Pipeline: latest unscaled features (live file if synced, else the processed file)
  → scale with the train-fitted scaler → last SEQ_LEN rows → model → scaled r̂
  → r̂ (log return) → P̂_{t+1} = P_t · exp(r̂).

The returned dict also carries the honest held-out test metrics of the model used and an
uncertainty band (± RMSE of the model's test-set return error), so the UI can never show a
number without its context.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from config import (PROCESSED_DATA_DIR, MODELS_DIR, SEQ_LEN, PREDICTION_HORIZON_DAYS, LEVEL_COLUMNS,
                    get_prefix, load_model_status)
from src.data.preprocessing import unscale_target
from src.models.registry import load_trained, TABULAR, RECURRENT
from src.models.ensemble_model import StackedModel, stack_path
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

DISCLAIMER = ("Research prototype. Next-day financial returns are close to unpredictable; on the held-out "
              "test period no model in this project beat the random-walk forecast by a statistically "
              "significant margin. This output is NOT financial advice.")


def _features_path(asset):
    p = get_prefix(asset)
    live = os.path.join(PROCESSED_DATA_DIR, f'{p}_live_features.csv')
    return live if os.path.exists(live) else os.path.join(PROCESSED_DATA_DIR, f'{p}_features.csv')


def load_latest_window(asset, seq_len=SEQ_LEN):
    """Returns (Xseq (1,seq,F), Xt (1,F), last_close, as_of_date, scaler, target_idx, columns)."""
    prefix = get_prefix(asset)
    feats = pd.read_csv(_features_path(asset), index_col='timestamp', parse_dates=True).sort_index()
    scaler = joblib.load(os.path.join(MODELS_DIR, f'{prefix}_scaler.pkl'))
    cols = [c for c in feats.columns if c not in LEVEL_COLUMNS]
    if len(cols) != scaler.n_features_in_:
        raise ValueError(f"Feature mismatch for {asset}: scaler expects {scaler.n_features_in_}, got {len(cols)}")
    scaled = scaler.transform(feats[cols].values)
    window = scaled[-seq_len:]
    Xseq = window[np.newaxis, :, :]
    return Xseq, window[-1:, :], float(feats['price'].iloc[-1]), feats.index[-1], scaler, cols.index('log_return'), cols


def available_models(asset):
    p = get_prefix(asset)
    out = []
    for m in list(TABULAR) + list(RECURRENT):
        f = {'CatBoost': f'{p}_catboost.cbm'}.get(m, f'{p}_{m.lower()}.pkl' if m in TABULAR else f'{p}_{m.lower()}.keras')
        if os.path.exists(os.path.join(MODELS_DIR, f)):
            out.append(m)
    if os.path.exists(stack_path(asset)):
        out.append('Stacked')
    return out


def _load(asset, model_name):
    return StackedModel(asset) if model_name == 'Stacked' else load_trained(model_name, asset)


def predict_next_day(asset, model_name=None):
    status = load_model_status().get(asset, {})
    model_name = model_name or status.get('primary_model', 'LightGBM')
    Xseq, Xt, last_close, as_of, scaler, ti, cols = load_latest_window(asset)
    model = _load(asset, model_name)
    r_hat = float(unscale_target(model.predict(Xseq, Xt), scaler, ti)[0])
    pred_price = last_close * np.exp(r_hat)

    test = status.get('test', {}) if model_name == status.get('primary_model') else {}
    band = test.get('RMSE_ret')                       # ±1 RMSE of held-out return error
    result = {
        'asset': asset, 'as_of_date': str(as_of.date()), 'horizon_days': PREDICTION_HORIZON_DAYS,
        'current_price': last_close, 'predicted_price': float(pred_price),
        'predicted_return_pct': r_hat * 100, 'direction': 'UP' if r_hat > 0 else 'DOWN',
        'model_used': model_name, 'is_served_model': model_name == status.get('primary_model'),
        'uncertainty_band': [float(last_close * np.exp(r_hat - band)), float(last_close * np.exp(r_hat + band))] if band else None,
        'test_metrics': test, 'test_naive': status.get('test_naive'), 'test_period': status.get('test_period'),
        'n_features': len(cols), 'disclaimer': DISCLAIMER,
    }
    logger.info(f"{asset}: {model_name} as of {as_of.date()} close ${last_close:,.2f} → ${pred_price:,.2f} ({r_hat*100:+.2f}%)")
    return result


def predict_all_models(asset):
    rows = []
    for m in available_models(asset):
        try:
            r = predict_next_day(asset, m)
            rows.append({'model': m, 'predicted_price': r['predicted_price'],
                         'predicted_return_pct': r['predicted_return_pct'], 'direction': r['direction']})
        except Exception as e:                      # one broken artefact must not kill the table
            logger.error(f"{asset}/{m}: {e}")
    return rows


def predict_next_day_safe(asset, model_name=None):
    try:
        return predict_next_day(asset, model_name)
    except Exception as e:
        logger.error(f"Failed to generate prediction for {asset}: {e}")
        return None


if __name__ == '__main__':
    from src.utils.logging_config import setup_logging
    setup_logging()
    for a in ('Bitcoin', 'Gold', 'Silver'):
        r = predict_next_day_safe(a)
        if r:
            print(f"{a:8s} {r['as_of_date']}  ${r['current_price']:,.2f} → ${r['predicted_price']:,.2f} "
                  f"({r['predicted_return_pct']:+.2f}%)  model={r['model_used']}")
