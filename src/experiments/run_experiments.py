"""
Logged experiments that justify the final design. Every experiment uses ONLY the train+val
period (walk-forward folds or the fixed validation window); the test set is never read.

    python src/experiments/run_experiments.py                # all experiments
    python src/experiments/run_experiments.py --only data features

Outputs (results/experiments/):
  E1_data_size.csv        3-year vs full-history training, same validation window
  E2_feature_ablation.csv drop one feature group at a time (walk-forward CV)
  E3_horizon.csv          next-day vs 5-day return target (walk-forward CV)
  E4_volatility.csv       22-day realised-volatility target vs persistence / EWMA baselines
  experiments_summary.md  human-readable summary of all of the above
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from config import ASSETS, EXPERIMENTS_DIR, CV_FOLDS
from src.data.preprocessing import build_dataset
from src.models.registry import make_model
from src.evaluation.cross_validation import cv_evaluate, summarise_folds
from src.utils.metrics import evaluate_task
from src.utils.logging_config import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)

TAB_MODELS = ['Ridge', 'LightGBM', 'CatBoost']
# Fixed, strongly regularised settings (the region the walk-forward tuner has consistently selected)
# so that every experiment compares like with like, independent of the tuner's progress.
EXP_PARAMS = {
    'Ridge': {'alpha': 100.0},
    'LightGBM': {'n_estimators': 600, 'learning_rate': 0.01, 'num_leaves': 4, 'min_child_samples': 50,
                 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 5.0},
    'CatBoost': {'iterations': 800, 'learning_rate': 0.01, 'depth': 3, 'l2_leaf_reg': 3.0},
}
P = lambda m, asset=None: EXP_PARAMS.get(m, {})
FEATURE_GROUPS = {
    'return path': ['log_return', 'return_1d', 'return_2d', 'return_5d', 'return_10d', 'return_20d'],
    'volatility state': ['volatility_10d', 'volatility_30d', 'rv_1d', 'rv_5d', 'rv_22d', 'ewma_vol', 'bb_width', 'atr_norm', 'hl_range'],
    'momentum / trend': ['RSI', 'ADX', 'ROC', 'ema14_ratio', 'ema30_ratio', 'macd_norm', 'macd_hist_norm', 'bb_pctb'],
    'macro': ['sp500_return', 'vix_return', 'dxy_return', 'oil_return', 'tnx_return', 'gold_return'],
    'sentiment / calendar / volume': ['fear_greed', 'dow_sin', 'dow_cos', 'log_volume_change'],
}


def fixed_val_eval(model_name, params, data):
    """Phase-A style: fit on train (early stopping on the last 15 % of train), score the fixed validation window."""
    n = len(data['y_train']); k = int(n * 0.85)
    m = make_model(model_name, params)
    m.fit(data['X_train'][:k], data['Xt_train'][:k], data['y_train'][:k],
          data['X_train'][k:], data['Xt_train'][k:], data['y_train'][k:], final=True)
    pred = data['inv'](m.predict(data['X_val'], data['Xt_val']))
    return evaluate_task(data, data['y_real_val'], pred, data['naive_val'], data['prev_val'])


def e1_data_size():
    rows = []
    for asset in ASSETS:
        for label, start in (('3 years (2023-08 →)', '2023-08-27'), ('full history (2018-01 →)', None)):
            d = build_dataset(asset, train_start=start)
            naive = evaluate_task(d, d['y_real_val'], d['naive_val'], d['naive_val'], d['prev_val'])
            rows.append({'asset': asset, 'training_data': label, 'n_train': len(d['y_train']), 'model': 'Naive',
                         'val_RMSE_ret': naive['RMSE_ret'], 'val_MAE_usd': naive['MAE_usd'], 'val_DirAcc': np.nan})
            for m in TAB_MODELS:
                r = fixed_val_eval(m, P(m), d)
                rows.append({'asset': asset, 'training_data': label, 'n_train': len(d['y_train']), 'model': m,
                             'val_RMSE_ret': r['RMSE_ret'], 'val_MAE_usd': r['MAE_usd'], 'val_DirAcc': r['DirAcc_pct']})
                log.info(f"E1 {asset:8s} {label:26s} {m:9s} n={len(d['y_train']):5d} val RMSE_ret={r['RMSE_ret']:.5f} DA={r['DirAcc_pct']:.1f}")
    df = pd.DataFrame(rows); df.to_csv(os.path.join(EXPERIMENTS_DIR, 'E1_data_size.csv'), index=False)
    return df


def e2_feature_ablation():
    rows = []
    for asset in ASSETS:
        d = build_dataset(asset)
        cols = d['columns']
        for m in ['Ridge', 'LightGBM']:
            params = P(m)
            base = summarise_folds(cv_evaluate(m, params, d, n_splits=CV_FOLDS))
            rows.append({'asset': asset, 'model': m, 'dropped_group': '(none — all features)', 'n_features': len(cols),
                         'cv_RMSE_ret': base['RMSE_ret_mean'], 'cv_RMSE_ret_std': base['RMSE_ret_std'], 'cv_DirAcc': base['DirAcc_pct_mean'],
                         'delta_RMSE_pct': 0.0})
            for g, feats in FEATURE_GROUPS.items():
                keep = [i for i, c in enumerate(cols) if c not in feats]
                if len(keep) == len(cols):
                    continue
                s = summarise_folds(cv_evaluate(m, params, d, n_splits=CV_FOLDS, feature_idx=keep))
                delta = (s['RMSE_ret_mean'] / base['RMSE_ret_mean'] - 1) * 100
                rows.append({'asset': asset, 'model': m, 'dropped_group': g, 'n_features': len(keep),
                             'cv_RMSE_ret': s['RMSE_ret_mean'], 'cv_RMSE_ret_std': s['RMSE_ret_std'], 'cv_DirAcc': s['DirAcc_pct_mean'],
                             'delta_RMSE_pct': delta})
                log.info(f"E2 {asset:8s} {m:9s} drop {g:30s} RMSE {s['RMSE_ret_mean']:.5f} ({delta:+.2f}%) DA {s['DirAcc_pct_mean']:.1f}")
    df = pd.DataFrame(rows); df.to_csv(os.path.join(EXPERIMENTS_DIR, 'E2_feature_ablation.csv'), index=False)
    return df


def e3_horizon():
    rows = []
    for asset in ASSETS:
        for task in ('return_1d', 'return_5d'):
            d = build_dataset(asset, task=task)
            for m in ['Naive', 'Naive-Mean'] + TAB_MODELS:
                s = summarise_folds(cv_evaluate(m, P(m), d, n_splits=CV_FOLDS))
                rows.append({'asset': asset, 'task': task, 'model': m, 'cv_RMSE_ret': s['RMSE_ret_mean'], 'cv_RMSE_ret_std': s['RMSE_ret_std'],
                             'cv_R2_ret': s['R2_ret_mean'], 'cv_DirAcc': s['DirAcc_pct_mean']})
                log.info(f"E3 {asset:8s} {task:10s} {m:11s} RMSE {s['RMSE_ret_mean']:.5f} R2 {s['R2_ret_mean']:+.3f} DA {s['DirAcc_pct_mean']:.1f}")
    df = pd.DataFrame(rows)
    # ratio to the naive forecast of the same task makes horizons comparable
    naive = df[df.model == 'Naive'].set_index(['asset', 'task'])['cv_RMSE_ret']
    df['RMSE_vs_naive_pct'] = [(r.cv_RMSE_ret / naive[(r.asset, r.task)] - 1) * 100 for r in df.itertuples()]
    df.to_csv(os.path.join(EXPERIMENTS_DIR, 'E3_horizon.csv'), index=False)
    return df


def e4_volatility():
    rows = []
    for asset in ASSETS:
        d = build_dataset(asset, task='vol_22d')
        for m in ['Naive', 'EWMA', 'Naive-Mean', 'Ridge', 'LightGBM', 'CatBoost']:
            s = summarise_folds(cv_evaluate(m, P(m), d, n_splits=CV_FOLDS))
            rows.append({'asset': asset, 'model': m, 'cv_RMSE_logvol': s['RMSE_log_mean'], 'cv_RMSE_logvol_std': s['RMSE_log_std'],
                         'cv_R2_logvol': s['R2_log_mean'], 'cv_QLIKE': s['QLIKE_mean'], 'cv_DirAcc_volchange': s['DirAcc_pct_mean']})
            log.info(f"E4 {asset:8s} {m:11s} RMSE_log {s['RMSE_log_mean']:.4f} R2 {s['R2_log_mean']:+.3f} QLIKE {s['QLIKE_mean']:.3f}")
    df = pd.DataFrame(rows)
    naive = df[df.model == 'Naive'].set_index('asset')['cv_RMSE_logvol']
    df['RMSE_vs_persistence_pct'] = [(r.cv_RMSE_logvol / naive[r.asset] - 1) * 100 for r in df.itertuples()]
    df.to_csv(os.path.join(EXPERIMENTS_DIR, 'E4_volatility.csv'), index=False)
    return df


def write_summary(res):
    L = ['# Experiment summary (validation only — the test set is never used here)', '']
    if 'data' in res:
        d = res['data']; L += ['## E1 — Training-data size (fixed validation window 2025-09-11 → 2026-02-17)', '',
                               '| Asset | Model | 3 y: val RMSE (ret) | full: val RMSE (ret) | change | 3 y DA | full DA |', '|---|---|---:|---:|---:|---:|---:|']
        for a in ASSETS:
            for m in ['Naive'] + TAB_MODELS:
                x = d[(d.asset == a) & (d.model == m)]
                r3, rf = x.iloc[0], x.iloc[1]
                L.append(f"| {a} | {m} | {r3.val_RMSE_ret:.5f} | {rf.val_RMSE_ret:.5f} | {(rf.val_RMSE_ret / r3.val_RMSE_ret - 1) * 100:+.2f}% | "
                         f"{'—' if np.isnan(r3.val_DirAcc) else f'{r3.val_DirAcc:.1f}'} | {'—' if np.isnan(rf.val_DirAcc) else f'{rf.val_DirAcc:.1f}'} |")
        L.append('')
    if 'features' in res:
        d = res['features']; L += ['## E2 — Feature-group ablation (walk-forward CV; positive change = dropping the group hurts)', '',
                                   '| Asset | Model | Dropped group | RMSE change | DA |', '|---|---|---|---:|---:|']
        for r in d.itertuples():
            L.append(f"| {r.asset} | {r.model} | {r.dropped_group} | {r.delta_RMSE_pct:+.2f}% | {r.cv_DirAcc:.1f} |")
        L.append('')
    if 'horizon' in res:
        d = res['horizon']; L += ['## E3 — Horizon: next-day vs 5-day return (walk-forward CV)', '',
                                  '| Asset | Task | Model | RMSE vs naive | R² (ret) | DA |', '|---|---|---|---:|---:|---:|']
        for r in d.itertuples():
            L.append(f"| {r.asset} | {r.task} | {r.model} | {r.RMSE_vs_naive_pct:+.2f}% | {r.cv_R2_ret:+.3f} | {r.cv_DirAcc:.1f} |")
        L.append('')
    if 'vol' in res:
        d = res['vol']; L += ['## E4 — 22-day realised-volatility target (walk-forward CV)', '',
                              '| Asset | Model | RMSE (log vol) | vs persistence | R² (log vol) | QLIKE | DA (vol change) |', '|---|---|---:|---:|---:|---:|---:|']
        for r in d.itertuples():
            L.append(f"| {r.asset} | {r.model} | {r.cv_RMSE_logvol:.4f} | {r.RMSE_vs_persistence_pct:+.1f}% | {r.cv_R2_logvol:+.3f} | {r.cv_QLIKE:.3f} | {r.cv_DirAcc_volchange:.1f} |")
        L.append('')
    with open(os.path.join(EXPERIMENTS_DIR, 'experiments_summary.md'), 'w') as f:
        f.write('\n'.join(L))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='+', default=['data', 'features', 'horizon', 'vol'])
    args = ap.parse_args()
    res = {}
    if 'data' in args.only:
        res['data'] = e1_data_size()
    if 'features' in args.only:
        res['features'] = e2_feature_ablation()
    if 'horizon' in args.only:
        res['horizon'] = e3_horizon()
    if 'vol' in args.only:
        res['vol'] = e4_volatility()
    write_summary(res)
    log.info(f"Experiments written to {EXPERIMENTS_DIR}")
