"""
Hyper-parameter tuning by expanding-window walk-forward validation.

    python src/training/tune_models.py                                  # served task, all assets, all models
    python src/training/tune_models.py --task vol_5d --models LightGBM Ridge --assets Bitcoin

Uses ONLY the train+val period (build_dataset splits; the tuner never reads X_test).
Every configuration's per-fold scores are logged to results/tuning/<asset>_<model>.csv,
and the winner (lowest mean fold RMSE in return space) is written to
results/tuning/best_params.json, which `config.get_params()` reads at training time.
"""
import os
import sys
import json
import time
import argparse
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from config import ASSETS, TUNING_DIR, BEST_PARAMS_PATH, DEFAULT_PARAMS, CV_FOLDS, DEFAULT_TASK, TASKS
from src.data.preprocessing import build_dataset
from src.evaluation.cross_validation import cv_evaluate, summarise_folds, primary_metric
from src.utils.logging_config import get_logger

from src.utils.logging_config import setup_logging
setup_logging()
log = get_logger(__name__)

# Small, deliberate grids: the data set is small (~500-900 training windows), so the
# search is over regularisation strength rather than capacity.
GRIDS = {
    'Ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
    'RandomForest': {'n_estimators': [300], 'max_depth': [3, 6], 'min_samples_leaf': [10, 30]},
    'LightGBM': {'n_estimators': [600], 'learning_rate': [0.01, 0.03],
                 'num_leaves': [4, 8, 15, 31], 'min_child_samples': [20, 50, 100],
                 'subsample': [0.8], 'colsample_bytree': [0.8], 'reg_lambda': [0.0, 5.0]},
    'CatBoost': {'iterations': [800], 'learning_rate': [0.01, 0.03], 'depth': [3, 5, 7], 'l2_leaf_reg': [3.0, 10.0]},
    'GRU': {'units': [32], 'dropout': [0.2, 0.4], 'learning_rate': [0.001],
            'batch_size': [64], 'epochs': [40], 'patience': [6]},
    'LSTM': {'units': [32], 'dropout': [0.2, 0.4], 'learning_rate': [0.001],
             'batch_size': [64], 'epochs': [40], 'patience': [6]},
}


def grid(model_name):
    g = GRIDS[model_name]
    keys = list(g)
    for combo in itertools.product(*[g[k] for k in keys]):
        yield dict(zip(keys, combo))


def tune(asset, model_name, data):
    rows, t0 = [], time.time()
    pm = primary_metric(data)
    for i, params in enumerate(grid(model_name)):
        folds = cv_evaluate(model_name, params, data, n_splits=CV_FOLDS)
        s = summarise_folds(folds)
        rows.append({'config_id': i, **{f'p_{k}': v for k, v in params.items()}, **s})
        log.info(f"{asset:8s} {model_name:12s} cfg {i:2d} {params}  {pm}={s[pm + '_mean']:.5f}±{s[pm + '_std']:.5f}  DA={s['DirAcc_pct_mean']:.1f}%")
    df = pd.DataFrame(rows).sort_values(pm + '_mean')
    suffix = '' if data['task'] == 'return_1d' else f"_{data['task']}"
    df.to_csv(os.path.join(TUNING_DIR, f'{asset.lower()}_{model_name.lower()}{suffix}.csv'), index=False)
    best_id = int(df.iloc[0]['config_id'])
    best = list(grid(model_name))[best_id]
    log.info(f"{asset} {model_name}: best {best} in {time.time() - t0:.0f}s")
    return best, df.iloc[0].to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assets', nargs='+', default=ASSETS)
    ap.add_argument('--models', nargs='+', default=list(GRIDS))
    ap.add_argument('--task', default=DEFAULT_TASK, choices=list(TASKS))
    args = ap.parse_args()

    best_params = json.load(open(BEST_PARAMS_PATH)) if os.path.exists(BEST_PARAMS_PATH) else {}
    suffix = '' if args.task == 'return_1d' else f'_{args.task}'
    summary_path = os.path.join(TUNING_DIR, f'tuning_summary{suffix}.csv')
    summary = pd.read_csv(summary_path).to_dict('records') if os.path.exists(summary_path) else []
    for asset in args.assets:
        data = build_dataset(asset, task=args.task)
        for m in args.models:
            best, score = tune(asset, m, data)
            best_params.setdefault(args.task, {}).setdefault(asset, {})[m] = best
            summary = [r for r in summary if not (r['asset'] == asset and r['model'] == m)]
            summary.append({'asset': asset, 'model': m, **{k: v for k, v in score.items() if k.endswith('_mean') or k.endswith('_std')}, 'best_params': json.dumps(best)})
            with open(BEST_PARAMS_PATH, 'w') as f:
                json.dump(best_params, f, indent=2)
            pd.DataFrame(summary).to_csv(summary_path, index=False)
    log.info(f"Best params written to {BEST_PARAMS_PATH}")


if __name__ == '__main__':
    main()
