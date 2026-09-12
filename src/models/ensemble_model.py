"""
Stacked ensemble (experiment).

Base models' out-of-fold predictions from the walk-forward folds (train+val only) are
used to fit a non-negative Ridge meta-model:

    r̂_stack = w_0 + Σ_k w_k · r̂_k        w_k ≥ 0

Non-negativity keeps the weights interpretable ("how much does the stack trust
model k?"). If the base models carry no out-of-sample signal, the weights shrink to
~0 and the stack degenerates to the intercept — that is a *result*, and the audit of
the original project found exactly this. The stack is therefore reported as an
experiment in the comparison table, not assumed to be the best model.

    python src/models/ensemble_model.py --assets Gold --bases Ridge LightGBM GRU
"""
import os
import sys
import json
import argparse
import numpy as np
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.linear_model import Ridge
from config import ASSETS, MODELS_DIR, RESULTS_DIR, get_params, get_prefix, CV_FOLDS
from src.data.preprocessing import build_dataset
from src.evaluation.cross_validation import cv_evaluate
from src.models.registry import load_trained
from src.utils.logging_config import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)
DEFAULT_BASES = ['Ridge', 'LightGBM', 'CatBoost', 'GRU']


def stack_path(asset):
    return os.path.join(MODELS_DIR, f'{get_prefix(asset)}_stack.pkl')


def fit_stack(asset, bases, data=None):
    data = data or build_dataset(asset)
    oof_cols, idx_ref = [], None
    for b in bases:
        _, oof, idx = cv_evaluate(b, get_params(b, asset, data['task']), data, n_splits=CV_FOLDS, return_oof=True)
        oof_cols.append(oof)
        idx_ref = idx
    y = np.concatenate([data['y_train'], data['y_val']])[idx_ref].ravel()
    X_meta = np.column_stack(oof_cols)
    meta = Ridge(alpha=1.0, positive=True).fit(X_meta, y)
    weights = dict(zip(bases, [float(w) for w in meta.coef_]))
    log.info(f"{asset}: stack weights {weights}  intercept={float(meta.intercept_):.4f}")
    joblib.dump({'meta': meta, 'bases': bases, 'weights': weights}, stack_path(asset))
    os.makedirs(os.path.join(RESULTS_DIR, 'stacking'), exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'stacking', f'{asset.lower()}_stack_weights.json'), 'w') as f:
        json.dump({'bases': bases, 'weights': weights, 'intercept': float(meta.intercept_)}, f, indent=2)
    return meta, weights


class StackedModel:
    """Inference wrapper: loads the meta-model and the deployed base models."""
    name = 'Stacked'

    def __init__(self, asset):
        obj = joblib.load(stack_path(asset))
        self.meta, self.bases, self.weights = obj['meta'], obj['bases'], obj['weights']
        self.base_models = [load_trained(b, asset) for b in self.bases]
        self.fit_info = {'weights': self.weights}

    def predict(self, Xseq, Xt):
        cols = [np.ravel(m.predict(Xseq, Xt)) for m in self.base_models]
        return self.meta.predict(np.column_stack(cols))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--assets', nargs='+', default=ASSETS)
    ap.add_argument('--bases', nargs='+', default=DEFAULT_BASES)
    args = ap.parse_args()
    for a in args.assets:
        fit_stack(a, args.bases)
