"""
Next-day prediction for the dashboard / API.

    from src.inference.prediction import predict_next_day, predict_for_date
    predict_next_day('Gold')                       # latest data, served model
    predict_next_day('Gold', 'GRU')                # any trained model
    predict_for_date('Gold', '2026-05-12')         # demo: predict the day AFTER 2026-05-12 using only
                                                   # data up to 2026-05-12, and compare with what happened

Pipeline: unscaled features up to day t → scale with the train-fitted scaler → last SEQ_LEN rows
  → model → standardised r̂ → r̂ (log return) → P̂_{t+1} = P_t · exp(r̂).

Every result carries the honest held-out test metrics of the model used and an uncertainty band
(± RMSE of the model's test-set return error), so the UI never shows a number without context.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from config import (PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, SEQ_LEN, PREDICTION_HORIZON_DAYS, LEVEL_COLUMNS,
                    VAL_END, get_prefix, load_model_status)
from src.models.registry import load_trained, artefact_exists, TABULAR, RECURRENT
from src.models.ensemble_model import StackedModel, stack_path
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

DISCLAIMER = ("Research prototype for an academic project. Next-day financial returns are close to unpredictable: "
              "read every forecast together with the held-out test metrics shown next to it (error size, direction "
              "hit-rate and the comparison with the random-walk forecast). This output is NOT financial advice.")


def _features_path(asset):
    p = get_prefix(asset)
    live = os.path.join(PROCESSED_DATA_DIR, f'{p}_live_features.csv')
    return live if os.path.exists(live) else os.path.join(PROCESSED_DATA_DIR, f'{p}_features.csv')


def load_features(asset):
    return pd.read_csv(_features_path(asset), index_col='timestamp', parse_dates=True).sort_index()


def _window(asset, feats, as_of=None, seq_len=SEQ_LEN):
    """Scaled window ending on `as_of` (default: last available day). Only rows <= as_of are used."""
    prefix = get_prefix(asset)
    scaler = joblib.load(os.path.join(MODELS_DIR, f'{prefix}_scaler.pkl'))
    cols = [c for c in feats.columns if c not in LEVEL_COLUMNS]
    if len(cols) != scaler.n_features_in_:
        raise ValueError(f"Feature mismatch for {asset}: scaler expects {scaler.n_features_in_}, got {len(cols)}")
    hist = feats if as_of is None else feats.loc[:pd.Timestamp(as_of)]
    if len(hist) < seq_len:
        raise ValueError(f"Not enough history before {as_of} ({len(hist)} rows, need {seq_len})")
    scaled = scaler.transform(hist[cols].values)
    window = scaled[-seq_len:]
    return window[np.newaxis, :, :], window[-1:, :], float(hist['price'].iloc[-1]), hist.index[-1], cols


def _target_stats(asset):
    """Mean/std used to standardise the training target (stored by backtesting in model_status)."""
    st = load_model_status().get(asset, {})
    return st.get('target_mean', 0.0), st.get('target_std', 1.0)


def available_models(asset):
    out = [m for m in list(TABULAR) + list(RECURRENT) if artefact_exists(m, asset)]
    if os.path.exists(stack_path(asset)):
        out.append('Stacked')
    return out


def _load(asset, model_name):
    return StackedModel(asset) if model_name == 'Stacked' else load_trained(model_name, asset)


def _predict_from_window(asset, model_name, Xseq, Xt):
    mu, sd = _target_stats(asset)
    return float(np.ravel(_load(asset, model_name).predict(Xseq, Xt))[0] * sd + mu)


def _context(asset, model_name, status):
    test = status.get('test', {}) if model_name == status.get('primary_model') else {}
    return {'model_used': model_name, 'is_served_model': model_name == status.get('primary_model'),
            'test_metrics': test, 'test_naive': status.get('test_naive'), 'test_period': status.get('test_period'),
            'horizon_days': PREDICTION_HORIZON_DAYS, 'disclaimer': DISCLAIMER}


def predict_next_day(asset, model_name=None):
    """Forecast for the trading day after the last available observation."""
    status = load_model_status().get(asset, {})
    model_name = model_name or status.get('primary_model', 'LightGBM')
    feats = load_features(asset)
    Xseq, Xt, last_close, as_of, cols = _window(asset, feats)
    r_hat = _predict_from_window(asset, model_name, Xseq, Xt)
    pred_price = last_close * np.exp(r_hat)
    band = (status.get('test') or {}).get('RMSE_ret') if model_name == status.get('primary_model') else None
    result = {
        'asset': asset, 'as_of_date': str(as_of.date()), 'current_price': last_close,
        'predicted_price': float(pred_price), 'predicted_return_pct': r_hat * 100,
        'direction': 'UP' if r_hat > 0 else 'DOWN', 'n_features': len(cols),
        'uncertainty_band': [float(last_close * np.exp(r_hat - band)), float(last_close * np.exp(r_hat + band))] if band else None,
        **_context(asset, model_name, status),
    }
    logger.info(f"{asset}: {model_name} as of {as_of.date()} close ${last_close:,.2f} → ${pred_price:,.2f} ({r_hat*100:+.2f}%)")
    return result


def predict_for_date(asset, as_of, model_name=None):
    """
    Demonstration mode: stand on day `as_of`, use ONLY data up to that day, predict the next
    trading day, then reveal the actual close (if the next day exists in the data) and the error.
    Days after config.VAL_END are outside the model's training data (unseen test period).
    """
    status = load_model_status().get(asset, {})
    model_name = model_name or status.get('primary_model', 'LightGBM')
    feats = load_features(asset)
    Xseq, Xt, last_close, as_of_ts, cols = _window(asset, feats, as_of)
    r_hat = _predict_from_window(asset, model_name, Xseq, Xt)
    pred_price = last_close * np.exp(r_hat)
    pos = feats.index.get_loc(as_of_ts)
    result = {
        'asset': asset, 'as_of_date': str(as_of_ts.date()), 'current_price': last_close,
        'predicted_price': float(pred_price), 'predicted_return_pct': r_hat * 100,
        'direction': 'UP' if r_hat > 0 else 'DOWN',
        'in_unseen_test_period': bool(as_of_ts >= pd.Timestamp(VAL_END)),
        'actual_date': None, 'actual_price': None, 'actual_return_pct': None, 'error_usd': None, 'error_pct': None, 'direction_hit': None,
        **_context(asset, model_name, status),
    }
    if pos + 1 < len(feats):
        actual = float(feats['price'].iloc[pos + 1])
        actual_ret = float(np.log(actual / last_close))
        result.update({'actual_date': str(feats.index[pos + 1].date()), 'actual_price': actual,
                       'actual_return_pct': actual_ret * 100, 'error_usd': float(pred_price - actual),
                       'error_pct': float((pred_price - actual) / actual * 100),
                       'direction_hit': bool(np.sign(r_hat) == np.sign(actual_ret)) if actual_ret != 0 else None})
    return result


def predict_all_models(asset, as_of=None):
    rows = []
    for m in available_models(asset):
        try:
            r = predict_for_date(asset, as_of, m) if as_of else predict_next_day(asset, m)
            rows.append({'model': m, 'predicted_price': r['predicted_price'], 'predicted_return_pct': r['predicted_return_pct'],
                         'direction': r['direction'], 'actual_price': r.get('actual_price'), 'error_pct': r.get('error_pct')})
        except Exception as e:                      # one broken artefact must not kill the table
            logger.error(f"{asset}/{m}: {e}")
    return rows


def prediction_history(asset):
    """The out-of-sample prediction log written by the evaluation script (one row per unseen day)."""
    path = os.path.join(RESULTS_DIR, 'predictions', f'{get_prefix(asset)}_test_predictions.csv')
    return pd.read_csv(path, parse_dates=['date', 'target_date']) if os.path.exists(path) else None


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
