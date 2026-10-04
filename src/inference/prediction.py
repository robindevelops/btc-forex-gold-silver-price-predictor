"""
Next-day prediction for the dashboard and the API.

    from src.inference.prediction import predict_next_day, predict_for_date
    predict_next_day('Gold')                       # latest complete bar → COMBINED forecast (default)
    predict_next_day('Gold', 'GRU')                # any single trained model
    predict_for_date('Gold', '2026-05-12')         # demo: predict the day AFTER 2026-05-12 using only
                                                   # data up to 2026-05-12, and compare with what happened

Pipeline: unscaled features up to day t → scale with the train-fitted scaler → last SEQ_LEN rows
  → model → standardised r̂ → r̂ (log return) → P̂_{t+1} = P_t · exp(r̂).

The served forecast is COMBINED: the equal-weight mean of the predicted log returns of every trained base
model (Ridge, RandomForest, LightGBM, CatBoost, GRU, LSTM). No weights are fitted, so nothing is selected on
the test set; the combination is evaluated on the same unseen days as every single model
(results/final_test_results.csv, row 'Combined'). The stacked ensemble is not a member because it is itself
a blend of some of these models. The user never has to choose a model.

Every result carries the held-out test metrics of the forecast used and a CONDITIONAL 68 % uncertainty band
    P_t · exp(r̂ ± k·σ_t),   σ_t = RiskMetrics EWMA volatility of the returns up to day t (the `ewma_vol` feature),
so the band widens in volatile regimes and narrows in calm ones. k is calibrated by src/evaluation/backtesting.py as the
68.27 % quantile of |r − r̂| / σ_t over the walk-forward out-of-fold errors of the TRAINING period (k ≈ 0.83–0.91: daily
returns are fat-tailed, so a plain ±1σ band covered ~73–75 % of days instead of 68 %). σ_t uses only information known at
day t; the band's coverage is reported on validation and test (results/band_calibration.csv). The UI never shows a number
without context.
"""
import os
import sys
import time
import datetime as dt
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from config import (PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, SEQ_LEN, PREDICTION_HORIZON_DAYS, LEVEL_COLUMNS,
                    VAL_END, ASSET_CONFIG, get_prefix, load_model_status)
from src.models.registry import load_trained, artefact_exists, TABULAR, RECURRENT
from src.models.ensemble_model import StackedModel, stack_path
from src.data.market_calendar import expected_last_complete_bar, next_trading_day
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

DISCLAIMER = ("Research prototype for an academic project. Next-day financial returns are close to unpredictable: "
              "read every forecast together with the held-out test metrics shown next to it (error size, direction "
              "hit-rate and the comparison with the random-walk forecast). This output is NOT financial advice.")

COMBINED = 'Combined'
BASE_MODELS = list(TABULAR) + list(RECURRENT)          # the members of the combined forecast, if trained
BAND_SIGMA_FEATURE = 'ewma_vol'                        # σ_t of the band: EWMA (λ = 0.94) volatility at day t
BAND_NOMINAL = 0.6827                                  # target coverage of the band (the probability mass of a ±1σ Gaussian band)
BAND_RULE = (f'68 % band P_t · exp(r̂ ± k·σ_t), σ_t = {BAND_SIGMA_FEATURE} at day t (RiskMetrics EWMA, λ = 0.94), '
             'k calibrated on walk-forward errors of the training period')


# ─────────────────────────────────────────────── data
def _features_path(asset, live=True):
    """Live download when present (next-day forecasts); the frozen evaluation dataset for the unseen-test demo."""
    p = get_prefix(asset)
    live_path = os.path.join(PROCESSED_DATA_DIR, f'{p}_live_features.csv')
    return live_path if (live and os.path.exists(live_path)) else os.path.join(PROCESSED_DATA_DIR, f'{p}_features.csv')


def load_features(asset, live=True):
    path = _features_path(asset, live)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No processed data for {asset} ({path}). Run `make preprocess` first.")
    return pd.read_csv(path, index_col='timestamp', parse_dates=True).sort_index()


def live_data_is_current(asset, now=None):
    """True when the live feature file already ends on the newest complete bar (no download needed)."""
    p = _features_path(asset)
    if not p.endswith('_live_features.csv'):
        return False
    try:
        last = pd.read_csv(p, usecols=['timestamp'], parse_dates=['timestamp'])['timestamp'].max().date()
    except Exception:
        return False
    return last >= expected_last_complete_bar(ASSET_CONFIG[asset]['type'], now)


def _window(asset, feats, as_of=None, seq_len=SEQ_LEN):
    """Scaled window ending on `as_of` (default: last available day). Only rows <= as_of are used."""
    prefix = get_prefix(asset)
    scaler = joblib.load(os.path.join(MODELS_DIR, f'{prefix}_scaler.pkl'))
    cols = [c for c in feats.columns if c not in LEVEL_COLUMNS]
    if len(cols) != scaler.n_features_in_:
        raise ValueError(f"Feature mismatch for {asset}: scaler expects {scaler.n_features_in_}, got {len(cols)}")
    expected = list(getattr(scaler, 'feature_names_in_', cols))
    if set(cols) != set(expected):
        raise ValueError(f"Feature mismatch for {asset}: {sorted(set(cols) ^ set(expected))}")
    cols = expected                                        # the scaler's (= the models') column order, whatever the file's order
    hist = feats if as_of is None else feats.loc[:pd.Timestamp(as_of)]
    if len(hist) < seq_len:
        raise ValueError(f"Not enough history before {as_of} ({len(hist)} rows, need {seq_len})")
    scaled = scaler.transform(hist[cols].values)
    window = scaled[-seq_len:]
    return window[np.newaxis, :, :], window[-1:, :], float(hist['price'].iloc[-1]), hist.index[-1], cols


def _band_k(status):
    """Calibrated band multiplier written by the evaluation script (model_status.json: band.k)."""
    k = (status.get('band') or {}).get('k')
    if k is None or not np.isfinite(k) or k <= 0:
        raise FileNotFoundError("model_status.json has no calibrated band multiplier: run `make evaluate` "
                                "(python src/evaluation/backtesting.py) before predicting")
    return float(k)


def _band(feats, as_of_ts, last_close, r_hat, k):
    """68 % price band for day t+1 from the volatility known at day t (unscaled feature value on row `as_of_ts`)."""
    sigma = float(feats.loc[as_of_ts, BAND_SIGMA_FEATURE])
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"invalid {BAND_SIGMA_FEATURE} on {as_of_ts.date()}: {sigma}")
    half = k * sigma
    return {'uncertainty_band': [float(last_close * np.exp(r_hat - half)), float(last_close * np.exp(r_hat + half))],
            'band_sigma_pct': sigma * 100, 'band_k': k, 'band_halfwidth_pct': half * 100, 'band_rule': BAND_RULE}


def _target_stats(asset):
    """Mean/std used to standardise the training target (stored by backtesting in model_status)."""
    st = load_model_status().get(asset, {})
    if 'target_mean' not in st or 'target_std' not in st:
        raise FileNotFoundError(f"model_status.json has no target statistics for {asset}: run `make evaluate` "
                                "(python src/evaluation/backtesting.py) before predicting")
    return st['target_mean'], st['target_std']


# ─────────────────────────────────────────────── models
def available_models(asset):
    """Every single model with a saved artefact (+ the stacked-ensemble experiment)."""
    out = [m for m in BASE_MODELS if artefact_exists(m, asset)]
    if os.path.exists(stack_path(asset)):
        out.append('Stacked')
    return out


def combined_members(asset):
    """The trained base models that enter the equal-weight combination."""
    return [m for m in BASE_MODELS if artefact_exists(m, asset)]


def _load(asset, model_name):
    return StackedModel(asset) if model_name == 'Stacked' else load_trained(model_name, asset)


def _predict_one(asset, model_name, Xseq, Xt):
    mu, sd = _target_stats(asset)
    r = float(np.ravel(_load(asset, model_name).predict(Xseq, Xt))[0] * sd + mu)
    if not np.isfinite(r):
        raise ValueError(f"{model_name} returned a non-finite prediction for {asset}")
    return r


def _predict_from_window(asset, model_name, Xseq, Xt, progress=None):
    """Real log-return prediction r̂ and, for the combined forecast, every member's r̂.
    `progress(model, r_hat, seconds)` is called after each member has run (the dashboard shows it live)."""
    if model_name != COMBINED:
        return _predict_one(asset, model_name, Xseq, Xt), None
    members = {}
    for m in combined_members(asset):
        t0 = time.perf_counter()
        members[m] = _predict_one(asset, m, Xseq, Xt)
        if progress:
            progress(m, members[m], time.perf_counter() - t0)
    if not members:
        raise FileNotFoundError(f"No trained models found for {asset}. Run `make train` first.")
    return float(np.mean(list(members.values()))), members


def _individual(members, last_close):
    if members is None:
        return None
    return [{'model': m, 'predicted_return_pct': r * 100, 'predicted_price': float(last_close * np.exp(r)),
             'direction': 'UP' if r > 0 else 'DOWN'} for m, r in members.items()]


def _context(asset, model_name, status):
    """Held-out test metrics for the forecast used: Combined and the CV-selected single model have them."""
    if model_name == COMBINED:
        test = status.get('combined_test') or {}
    elif model_name == status.get('primary_model'):
        test = status.get('test') or {}
    else:
        test = {}
    return {'model_used': model_name, 'is_served_model': model_name == COMBINED,
            'cv_selected_model': status.get('primary_model'), 'test_metrics': test,
            'test_naive': status.get('test_naive'), 'test_period': status.get('test_period'),
            'horizon_days': PREDICTION_HORIZON_DAYS, 'disclaimer': DISCLAIMER}


# ─────────────────────────────────────────────── public API
def predict_next_day(asset, model_name=COMBINED):
    """Forecast for the trading day after the last complete bar (default: the combined forecast)."""
    status = load_model_status().get(asset, {})
    model_name = model_name or COMBINED
    feats = load_features(asset)
    Xseq, Xt, last_close, as_of, cols = _window(asset, feats)
    r_hat, members = _predict_from_window(asset, model_name, Xseq, Xt)
    pred_price = last_close * np.exp(r_hat)
    ctx = _context(asset, model_name, status)
    result = {
        'asset': asset, 'as_of_date': str(as_of.date()), 'current_price': last_close,
        'target_date': str(next_trading_day(ASSET_CONFIG[asset]['type'], as_of)),
        'predicted_price': float(pred_price), 'predicted_return_pct': r_hat * 100,
        'direction': 'UP' if r_hat > 0 else 'DOWN', 'n_features': len(cols),
        **_band(feats, as_of, last_close, r_hat, _band_k(status)), 'band_calibration': status.get('band'),
        'data_source': 'live download' if _features_path(asset).endswith('_live_features.csv') else 'stored dataset',
        'individual': _individual(members, last_close), **ctx,
    }
    logger.info(f"{asset}: {model_name} as of {as_of.date()} close ${last_close:,.2f} → ${pred_price:,.2f} ({r_hat*100:+.2f}%)")
    return result


def predict_for_date(asset, as_of, model_name=COMBINED, progress=None):
    """
    Demonstration mode on the FROZEN dataset: stand on day `as_of`, use ONLY data up to that day, predict the
    next trading day, then reveal the actual close (if the next day exists in the data) and the error.
    Days after config.VAL_END are outside the model's training data (unseen test period).
    """
    status = load_model_status().get(asset, {})
    model_name = model_name or COMBINED
    feats = load_features(asset, live=False)              # the frozen dataset the test results were computed on
    t_start = time.perf_counter()
    Xseq, Xt, last_close, as_of_ts, cols = _window(asset, feats, as_of)
    r_hat, members = _predict_from_window(asset, model_name, Xseq, Xt, progress)
    compute_seconds = time.perf_counter() - t_start
    pred_price = last_close * np.exp(r_hat)
    pos = feats.index.get_loc(as_of_ts)
    result = {
        'asset': asset, 'as_of_date': str(as_of_ts.date()), 'current_price': last_close,
        'predicted_price': float(pred_price), 'predicted_return_pct': r_hat * 100,
        'direction': 'UP' if r_hat > 0 else 'DOWN',
        'in_unseen_test_period': bool(as_of_ts >= pd.Timestamp(VAL_END)),
        **_band(feats, as_of_ts, last_close, r_hat, _band_k(status)), 'band_calibration': status.get('band'),
        'actual_date': None, 'actual_price': None, 'actual_return_pct': None, 'error_usd': None, 'error_pct': None, 'direction_hit': None,
        'in_band': None, 'individual': _individual(members, last_close), **_context(asset, model_name, status),
        # provenance of this run: computed now, from data up to as_of only
        'computed_at': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'compute_seconds': compute_seconds,
        'rows_used': int((feats.index <= as_of_ts).sum()), 'rows_hidden': int((feats.index > as_of_ts).sum()),
        'history_start': str(feats.index[0].date()), 'window_start': str(feats.index[max(pos - SEQ_LEN + 1, 0)].date()),
    }
    if pos + 1 < len(feats):
        actual = float(feats['price'].iloc[pos + 1])
        actual_ret = float(np.log(actual / last_close))
        lo, hi = result['uncertainty_band']
        result.update({'actual_date': str(feats.index[pos + 1].date()), 'actual_price': actual,
                       'actual_return_pct': actual_ret * 100, 'error_usd': float(pred_price - actual),
                       'error_pct': float((pred_price - actual) / actual * 100),
                       'direction_hit': bool(np.sign(r_hat) == np.sign(actual_ret)) if actual_ret != 0 else None,
                       'in_band': bool(lo <= actual <= hi)})
        for row in result['individual'] or []:
            row['error_pct'] = (row['predicted_price'] - actual) / actual * 100
            row['direction_hit'] = (row['direction'] == ('UP' if actual_ret > 0 else 'DOWN')) if actual_ret != 0 else None
    return result


def recent_track_record(asset, n_days=30, progress=None):
    """
    Live scorecard for the dashboard: every trained model is re-run NOW on each of the last `n_days` complete bars of the
    current data (live download when present) — for day t it sees only the window ending on t — and its forecast is scored
    against the actual close of day t+1. Nothing is read from the stored results.

    Returns {'rows': one dict per model + Combined + Random walk, 'days': per-day DataFrame (Combined), plus provenance}.
    `progress(model, seconds)` is called after each model has scored all days.
    """
    t_start = time.perf_counter()
    feats = load_features(asset)
    scaler = joblib.load(os.path.join(MODELS_DIR, f'{get_prefix(asset)}_scaler.pkl'))
    cols = list(getattr(scaler, 'feature_names_in_', [c for c in feats.columns if c not in LEVEL_COLUMNS]))
    missing = set(cols) - set(feats.columns)
    if missing:
        raise ValueError(f"Feature mismatch for {asset}: missing {sorted(missing)}")
    scaled = scaler.transform(feats[cols].values)
    price = feats['price'].values
    n_days = int(min(n_days, len(feats) - SEQ_LEN - 1))
    ends = np.arange(len(feats) - 1 - n_days, len(feats) - 1)            # day t; day t+1 exists for every one of them
    Xseq = np.stack([scaled[e - SEQ_LEN + 1:e + 1] for e in ends])
    Xt = scaled[ends]
    prev, nxt = price[ends], price[ends + 1]
    actual = np.log(nxt / prev)
    mu, sd = _target_stats(asset)
    preds = {}
    for m in combined_members(asset):
        t0 = time.perf_counter()
        preds[m] = np.ravel(_load(asset, m).predict(Xseq, Xt)) * sd + mu
        if not np.all(np.isfinite(preds[m])):
            raise ValueError(f"{m} returned a non-finite prediction for {asset}")
        if progress:
            progress(m, time.perf_counter() - t0)
    if not preds:
        raise FileNotFoundError(f"No trained models found for {asset}. Run `make train` first.")
    preds[COMBINED] = np.mean(list(preds.values()), axis=0)
    moved = actual != 0
    rw_err = np.abs(prev / nxt - 1) * 100                                 # "tomorrow = today"
    rows = []
    for m, r in preds.items():
        err = np.abs(prev * np.exp(r) / nxt - 1) * 100
        rows.append({'model': m, 'n_days': len(ends),
                     'hits': int(np.sum(np.sign(r[moved]) == np.sign(actual[moved]))), 'n_moved': int(moved.sum()),
                     'hit_pct': float(np.mean(np.sign(r[moved]) == np.sign(actual[moved])) * 100) if moved.any() else float('nan'),
                     'mae_pct': float(err.mean()), 'rw_mae_pct': float(rw_err.mean()),
                     'better_days_pct': float(np.mean(err < rw_err) * 100), 'up_calls_pct': float(np.mean(r > 0) * 100)})
    rows.append({'model': 'Random walk', 'n_days': len(ends), 'hits': None, 'n_moved': int(moved.sum()), 'hit_pct': float('nan'),
                 'mae_pct': float(rw_err.mean()), 'rw_mae_pct': float(rw_err.mean()), 'better_days_pct': float('nan'),
                 'up_calls_pct': float('nan')})
    dates = feats.index[ends]
    rc = preds[COMBINED]
    days = pd.DataFrame({'date': dates.date, 'target_date': feats.index[ends + 1].date, 'close': prev,
                         'predicted': prev * np.exp(rc), 'actual': nxt, 'predicted_pct': rc * 100, 'actual_pct': actual * 100,
                         'hit': np.where(~moved, None, np.sign(rc) == np.sign(actual))})
    frozen_end = load_features(asset, live=False).index[-1]
    return {'rows': rows, 'days': days, 'first_day': str(dates[0].date()), 'last_day': str(dates[-1].date()),
            'last_target': str(feats.index[ends[-1] + 1].date()), 'n_days': len(ends),
            'n_after_training': int(np.sum(dates >= pd.Timestamp(VAL_END))),
            'n_after_frozen_test': int(np.sum(feats.index[ends + 1] > frozen_end)),
            'data_source': 'live download' if _features_path(asset).endswith('_live_features.csv') else 'stored dataset',
            'computed_at': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'compute_seconds': time.perf_counter() - t_start}


def prediction_history(asset):
    """The out-of-sample prediction log written by the evaluation script (one row per unseen test day)."""
    path = os.path.join(RESULTS_DIR, 'predictions', f'{get_prefix(asset)}_test_predictions.csv')
    return pd.read_csv(path, parse_dates=['date', 'target_date']) if os.path.exists(path) else None


def predict_next_day_safe(asset, model_name=COMBINED):
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
            members = ', '.join(f"{m['model']} {m['predicted_return_pct']:+.2f}%" for m in r['individual'])
            print(f"{a:8s} {r['as_of_date']}  ${r['current_price']:,.2f} → ${r['predicted_price']:,.2f} "
                  f"({r['predicted_return_pct']:+.2f}% {r['direction']})  68% band ${r['uncertainty_band'][0]:,.2f}–${r['uncertainty_band'][1]:,.2f} "
                  f"(±{r['band_halfwidth_pct']:.2f}% = {r['band_k']:.2f}·σ_t)  combined of [{members}]")
