"""
Final-audit improvement candidates (2026-10-04) — decided on the walk-forward folds ONLY (train+val; the test set is never read).

    python src/experiments/audit_improvements.py          → results/experiments/E5_audit_candidates.csv (+ E5_summary.md)

Every candidate is scored on the same pooled out-of-fold days as the served Combined forecast (≈ 6 years per asset) and
kept only if it lowers the pooled RMSE for every asset with a Diebold–Mariano p < 0.05 against the served forecast.

  E5a  membership   Combined of the 4 tabular models only (drops GRU/LSTM, which are the weakest members for Bitcoin)
  E5b  features     tabular models on a de-duplicated feature set: drop the 10 features with |ρ| > 0.85 to a kept one
                    (ema14/ema30 ratio, %B, ROC, MACD/price ≈ RSI / 20-day return; 10/30-day vol, rv_5d, rv_22d, ATR/price ≈ EWMA vol)
  E5c  zero drift   served Combined minus the training-window mean return (forecast centred on the random walk)

Two further audit changes are NOT candidates here because they are part of the evaluation itself (src/evaluation/backtesting.py):
the pooled walk-forward significance test (results/walkforward_pooled.csv) and the calibrated 68 % band (results/band_calibration.csv).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from config import ASSETS, EXPERIMENTS_DIR, CV_FOLDS, get_params
from src.data.preprocessing import build_dataset
from src.evaluation.cross_validation import cv_evaluate, trainval_arrays
from src.models.registry import TABULAR, RECURRENT
from src.utils.metrics import diebold_mariano, directional_accuracy, rmse
from src.utils.logging_config import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)
ML = list(TABULAR) + list(RECURRENT)
REDUNDANT = ['ema14_ratio', 'ema30_ratio', 'bb_pctb', 'ROC', 'macd_norm',
             'volatility_10d', 'volatility_30d', 'rv_22d', 'atr_norm', 'rv_5d']


def run_asset(asset):
    d = build_dataset(asset)
    A = trainval_arrays(d)
    oof, idx = {}, None
    for m in ML:
        _, o, idx = cv_evaluate(m, get_params(m, asset), d, n_splits=CV_FOLDS, return_oof=True)
        oof[m] = d['inv'](o)
    keep = [i for i, c in enumerate(d['columns']) if c not in REDUNDANT]
    compact = {m: d['inv'](cv_evaluate(m, get_params(m, asset), d, n_splits=CV_FOLDS, return_oof=True, feature_idx=keep)[1])
               for m in TABULAR}
    y = A['y_real'][idx]
    # training-window mean return of each fold (TimeSeriesSplit validation blocks are equal-sized and consecutive)
    drift = np.empty(len(idx))
    for b in np.array_split(np.arange(len(idx)), CV_FOLDS):
        drift[b] = A['y_real'][:idx[b[0]]].mean()
    served = np.mean([oof[m] for m in ML], axis=0)
    cands = {'served: Combined (6 models, all features)': served,
             'E5a: Combined (4 tabular models)': np.mean([oof[m] for m in TABULAR], axis=0),
             'E5b: Combined (4 tabular models, compact features)': np.mean([compact[m] for m in TABULAR], axis=0),
             'E5c: Combined zero-drift': served - drift}
    rows, zero = [], np.zeros(len(y))
    for name, p in cands.items():
        _, p_naive = diebold_mariano(y, p, zero)
        _, p_served = diebold_mariano(y, p, served)
        da, n, p_da = directional_accuracy(y, p)
        rows.append({'asset': asset, 'candidate': name, 'n_features': len(keep) if 'compact' in name else len(d['columns']),
                     'n_days': len(y), 'RMSE_ret': rmse(y, p), 'RMSE_vs_naive_pct': (rmse(y, p) / rmse(y, zero) - 1) * 100,
                     'RMSE_vs_served_pct': (rmse(y, p) / rmse(y, served) - 1) * 100, 'DM_p_vs_naive': p_naive,
                     'DM_p_vs_served': (np.nan if name.startswith('served') else p_served), 'DirAcc_pct': da, 'DirAcc_pvalue': p_da,
                     'UpCalls_pct': float(np.mean(p > 0) * 100)})
    # E5b in isolation: compact vs full features for each tabular model
    for m in TABULAR:
        rows.append({'asset': asset, 'candidate': f'E5b detail: {m} compact vs full', 'n_features': len(keep), 'n_days': len(y),
                     'RMSE_ret': rmse(y, compact[m]), 'RMSE_vs_naive_pct': (rmse(y, compact[m]) / rmse(y, zero) - 1) * 100,
                     'RMSE_vs_served_pct': (rmse(y, compact[m]) / rmse(y, oof[m]) - 1) * 100,   # here: vs the same model on all features
                     'DM_p_vs_naive': diebold_mariano(y, compact[m], zero)[1], 'DM_p_vs_served': diebold_mariano(y, compact[m], oof[m])[1],
                     'DirAcc_pct': directional_accuracy(y, compact[m])[0], 'DirAcc_pvalue': directional_accuracy(y, compact[m])[2],
                     'UpCalls_pct': float(np.mean(compact[m] > 0) * 100)})
    return rows


def main():
    rows = []
    for a in ASSETS:
        rows += run_asset(a)
        log.info(f"{a}: done")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EXPERIMENTS_DIR, 'E5_audit_candidates.csv'), index=False)
    lines = ["# E5 — final-audit improvement candidates (walk-forward folds only; the test set is never read)", "",
             "Decision rule: keep a candidate only if it lowers the pooled out-of-fold RMSE for every asset with DM p < 0.05 against the served forecast.", "",
             "| Asset | Candidate | RMSE vs naive | RMSE vs served | DM p vs served | Dir. Acc % (p) | UP calls % |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in df[~df['candidate'].str.startswith('E5b detail')].iterrows():
        dms = '—' if pd.isna(r['DM_p_vs_served']) else f"{r['DM_p_vs_served']:.3f}"
        lines.append(f"| {r['asset']} | {r['candidate']} | {r['RMSE_vs_naive_pct']:+.3f} % | {r['RMSE_vs_served_pct']:+.3f} % | {dms} | "
                     f"{r['DirAcc_pct']:.1f} ({r['DirAcc_pvalue']:.3f}) | {r['UpCalls_pct']:.0f} |")
    lines += ["", "E5b detail (each tabular model, compact vs all features; positive = compact is worse):", "",
              "| Asset | Model | RMSE change | DM p |", "|---|---|---:|---:|"]
    for _, r in df[df['candidate'].str.startswith('E5b detail')].iterrows():
        lines.append(f"| {r['asset']} | {r['candidate'].split(': ')[1].split(' ')[0]} | {r['RMSE_vs_served_pct']:+.3f} % | {r['DM_p_vs_served']:.3f} |")
    with open(os.path.join(EXPERIMENTS_DIR, 'E5_summary.md'), 'w') as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == '__main__':
    main()
