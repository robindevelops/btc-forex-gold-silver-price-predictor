"""
Expanding-window walk-forward validation.

The train+val period is cut into CV_FOLDS consecutive validation blocks; for fold k the
model is trained on everything BEFORE the block and evaluated on the block. This mimics
how the model would actually be used (always predicting the future from the past) and
gives several independent out-of-sample estimates instead of one.

    |---------- train ----------|-- val 1 --|
    |---------------- train ----------------|-- val 2 --|
    |------------------------ train --------------------|-- val 3 --|   ...

The TEST split is never passed to anything in this module.
"""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from config import CV_FOLDS
from src.models.registry import make_model
from src.data.preprocessing import unscale_target
from src.utils.metrics import evaluate_forecast


def walk_forward_splits(n_samples, n_splits=CV_FOLDS, min_train=None):
    """Yield (train_idx, val_idx) with growing training windows and equal validation blocks."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr, va in tscv.split(np.arange(n_samples)):
        if min_train and len(tr) < min_train:
            continue
        yield tr, va


def _trainval_arrays(data):
    """Concatenate train and val (chronological) — the only data CV is allowed to see."""
    Xs = np.concatenate([data['X_train'], data['X_val']])
    Xt = np.concatenate([data['Xt_train'], data['Xt_val']])
    y = np.concatenate([data['y_train'], data['y_val']])
    prev = np.concatenate([data['prev_train'], data['prev_val']])
    return Xs, Xt, y, prev


def cv_evaluate(model_name, params, data, n_splits=CV_FOLDS, inner_val_frac=0.15, return_oof=False):
    """
    Walk-forward evaluation of one (model, params) on train+val.
    Inside each fold, the last `inner_val_frac` of the fold's training window is used as
    the early-stopping monitor for models that need one; the fold's validation block is
    only ever predicted, never fitted on.
    Returns a list of per-fold metric dicts (and OOF arrays if requested).
    """
    Xs, Xt, y, prev = _trainval_arrays(data)
    scaler, ti = data['scaler'], data['target_idx']
    zero_scaled = float(-scaler.data_min_[ti] / (scaler.data_max_[ti] - scaler.data_min_[ti]))
    folds, oof_pred, oof_idx = [], [], []

    for k, (tr, va) in enumerate(walk_forward_splits(len(y), n_splits)):
        n_inner = int(len(tr) * inner_val_frac)
        tr_fit, tr_mon = tr[:-n_inner], tr[-n_inner:]

        if model_name == 'ARIMA':
            m = make_model('ARIMA').fit_series(np.ravel(y[tr]))
            pred = m.forecast_walk_forward(np.ravel(y[va]))
        elif model_name == 'Naive-Zero':
            pred = np.full(len(va), zero_scaled)
        else:
            m = make_model(model_name, params)
            m.fit(Xs[tr_fit], Xt[tr_fit], y[tr_fit], Xs[tr_mon], Xt[tr_mon], y[tr_mon], final=True)
            pred = m.predict(Xs[va], Xt[va])

        met = evaluate_forecast(unscale_target(y[va], scaler, ti), unscale_target(pred, scaler, ti), prev[va])
        met.update({'fold': k + 1, 'n_train': len(tr), 'n_val': len(va)})
        folds.append(met)
        oof_pred.append(np.ravel(pred)); oof_idx.append(va)

    if return_oof:
        return folds, np.concatenate(oof_pred), np.concatenate(oof_idx)
    return folds


def summarise_folds(folds, keys=('RMSE_ret', 'MAE_ret', 'R2_ret', 'DirAcc_pct', 'RMSE_usd')):
    out = {}
    for k in keys:
        vals = np.array([f[k] for f in folds], float)
        out[f'{k}_mean'] = float(np.nanmean(vals))
        out[f'{k}_std'] = float(np.nanstd(vals))
    return out
