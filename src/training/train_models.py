"""
Train the final models for every asset.

    python src/training/train_models.py                      # served task, all assets, all models
    python src/training/train_models.py --task vol_5d --assets Bitcoin --models LightGBM GRU

For each (asset, model) two fits are made with the tuned hyper-parameters
(results/tuning/best_params.json, falling back to config.DEFAULT_PARAMS):

  Phase A  fit on TRAIN, monitor VAL      → honest validation metrics, train-vs-val
                                             over-fitting gap, Keras loss curves
  Phase B  refit on TRAIN+VAL             → the deployed model (data/models/*)
           (epoch / tree count fixed from phase A)

The TEST split is never read here. Outputs:
  data/models/<prefix>_<model>.{pkl|cbm|keras}     deployed models
  results/train_val_metrics.csv                     phase-A metrics for every model
  results/feature_importance.csv                    for models that expose it
  results/histories/<prefix>_<model>.json           loss curves
"""
import os
import sys
import json
import time
import argparse
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from config import ASSETS, RESULTS_DIR, get_params, DEFAULT_TASK, TASKS
from src.data.preprocessing import build_dataset
from src.models.registry import make_model, TABULAR, RECURRENT
from src.utils.metrics import evaluate_task
from src.utils.logging_config import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)
TRAINABLE = list(TABULAR) + list(RECURRENT)


def train_asset(asset, model_names, task=DEFAULT_TASK):
    data = build_dataset(asset, task=task)
    cols = data['columns']
    rows, imps = [], []
    for name in model_names:
        params = get_params(name, asset, task)
        t0 = time.time()
        # Phase A — honest validation fit
        m = make_model(name, params)
        m.fit(data['X_train'], data['Xt_train'], data['y_train'],
              data['X_val'], data['Xt_val'], data['y_val'], final=False)
        for split in ('train', 'val'):
            pred = m.predict(data[f'X_{split}'], data[f'Xt_{split}'])
            met = evaluate_task(data, data[f'y_real_{split}'], data['inv'](pred), data[f'naive_{split}'], data[f'prev_{split}'])
            rows.append({'asset': asset, 'model': name, 'task': task, 'split': split, **met, 'params': json.dumps(params),
                         **{f'fit_{k}': v for k, v in m.fit_info.items()}})
        if m.history:
            os.makedirs(os.path.join(RESULTS_DIR, 'histories'), exist_ok=True)
            suffix = '' if task == 'return_1d' else f'_{task}'
            with open(os.path.join(RESULTS_DIR, 'histories', f'{asset.lower()}_{name.lower()}{suffix}.json'), 'w') as f:
                json.dump(m.history, f)
        # Phase B — deployed model on train+val
        m_final = make_model(name, params)
        m_final.fit(data['X_train'], data['Xt_train'], data['y_train'],
                    data['X_val'], data['Xt_val'], data['y_val'], final=True)
        m_final.history = m.history
        path = m_final.save(asset, task)
        fi = m_final.feature_importance(cols)
        if fi:
            imps += [{'asset': asset, 'model': name, 'task': task, 'feature': k, 'importance': float(v)} for k, v in fi.items()]
        pm = 'RMSE_log' if data['kind'] == 'vol' else 'RMSE_ret'
        va = [r for r in rows if r['asset'] == asset and r['model'] == name and r['split'] == 'val'][0]
        tr = [r for r in rows if r['asset'] == asset and r['model'] == name and r['split'] == 'train'][0]
        log.info(f"{asset:8s} {name:12s} train {pm}={tr[pm]:.5f}  val {pm}={va[pm]:.5f}  "
                 f"val DA={va['DirAcc_pct']:.1f}%  {m_final.fit_info}  ({time.time() - t0:.0f}s) → {os.path.basename(path)}")
    return rows, imps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assets', nargs='+', default=ASSETS)
    ap.add_argument('--models', nargs='+', default=TRAINABLE)
    ap.add_argument('--task', default=DEFAULT_TASK, choices=list(TASKS))
    args = ap.parse_args()

    all_rows, all_imps = [], []
    for asset in args.assets:
        r, i = train_asset(asset, args.models, args.task)
        all_rows += r; all_imps += i

    suffix = '' if args.task == 'return_1d' else f'_{args.task}'
    tv_path = os.path.join(RESULTS_DIR, f'train_val_metrics{suffix}.csv')
    fi_path = os.path.join(RESULTS_DIR, f'feature_importance{suffix}.csv')
    # merge with existing rows for other assets/models so partial runs don't lose results
    for path, new, keys in ((tv_path, all_rows, ['asset', 'model']), (fi_path, all_imps, ['asset', 'model'])):
        df_new = pd.DataFrame(new)
        if os.path.exists(path) and len(df_new):
            old = pd.read_csv(path)
            old = old[~old.set_index(keys).index.isin(df_new.set_index(keys).index)]
            df_new = pd.concat([old, df_new], ignore_index=True)
        if len(df_new):
            df_new.to_csv(path, index=False)
    log.info(f"Saved {tv_path} and {fi_path}")


if __name__ == '__main__':
    main()
