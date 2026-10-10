"""
Submission-audit improvement candidates (2026-10-10) — decided on the walk-forward folds ONLY (train+val; the test set is never read).

    python src/experiments/final_candidates.py          → results/experiments/E6_final_candidates.csv (+ E6_summary.md)

Both candidates target the same weakness: daily returns are fat-tailed and heteroskedastic (Bitcoin −46 % on 2020-03-12, silver −37.6 %
on 2026-01-30), so a squared-error fit is dominated by a handful of turbulent days.

  E6a  volatility-scaled target   each tabular member learns z = r_{t+1} / σ_t (σ_t = EWMA volatility at day t, a feature known at t)
                                  and predicts r̂ = ẑ·σ_t — weighted least squares, the textbook fix for heteroskedastic noise
  E6b  robust loss                LightGBM and CatBoost trained with the Huber loss (δ = 1 training-target standard deviation)

Decision rule (fixed before running, same as E5): a candidate replaces the served forecast only if it lowers the pooled out-of-fold
RMSE for EVERY asset with a Diebold–Mariano p < 0.05 against the served Combined forecast. Otherwise nothing changes.
Candidates are scored on exactly the same pooled out-of-fold days as the served forecast (same folds, same early-stopping monitor).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from config import ASSETS, EXPERIMENTS_DIR, CV_FOLDS, get_params
from src.data.preprocessing import build_dataset
from src.evaluation.cross_validation import cv_evaluate, trainval_arrays, walk_forward_splits
from src.models.registry import make_model, TABULAR, RECURRENT
from src.utils.metrics import diebold_mariano, directional_accuracy, rmse
from src.utils.logging_config import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)
ML = list(TABULAR) + list(RECURRENT)
HUBER = {'LightGBM': {'objective': 'huber', 'alpha': 1.0}, 'CatBoost': {'loss_function': 'Huber:delta=1.0'}}


def vol_scaled_oof(model_name, params, d, inner_val_frac=0.15):
    """cv_evaluate's folds and early-stopping monitor, but the model learns z = r/σ_t (standardised with the training split's
    statistics, exactly like the served target) and the forecast is r̂ = ẑ·σ_t. Returns (oof real-return predictions, oof index)."""
    A = trainval_arrays(d)
    days = np.concatenate([d['dates_train'], d['dates_val']])
    sigma = d['features']['ewma_vol'].reindex(days).values
    z = A['y_real'] / sigma
    n_tr = len(d['y_train'])
    mu, sd = float(z[:n_tr].mean()), float(z[:n_tr].std())
    zs = ((z - mu) / sd).reshape(-1, 1)
    preds, idx = [], []
    for tr, va in walk_forward_splits(len(zs), CV_FOLDS):
        n_inner = int(len(tr) * inner_val_frac)
        tr_fit, tr_mon = tr[:-n_inner], tr[-n_inner:]
        m = make_model(model_name, params)
        m.fit(A['X'][tr_fit], A['Xt'][tr_fit], zs[tr_fit], A['X'][tr_mon], A['Xt'][tr_mon], zs[tr_mon], final=True)
        preds.append((np.ravel(m.predict(A['X'][va], A['Xt'][va])) * sd + mu) * sigma[va])
        idx.append(va)
    return np.concatenate(preds), np.concatenate(idx)


def score(name, p, y, served, zero):
    _, p_naive = diebold_mariano(y, p, zero)
    _, p_served = diebold_mariano(y, p, served)
    da, n, p_da = directional_accuracy(y, p)
    return {'candidate': name, 'n_days': len(y), 'RMSE_ret': rmse(y, p), 'RMSE_vs_naive_pct': (rmse(y, p) / rmse(y, zero) - 1) * 100,
            'RMSE_vs_served_pct': (rmse(y, p) / rmse(y, served) - 1) * 100, 'DM_p_vs_naive': p_naive,
            'DM_p_vs_served': (np.nan if name.startswith('served') else p_served), 'DirAcc_pct': da, 'DirAcc_pvalue': p_da,
            'UpCalls_pct': float(np.mean(p > 0) * 100)}


def run_asset(asset):
    d = build_dataset(asset)
    y_all = trainval_arrays(d)['y_real']
    oof, idx = {}, None
    for m in ML:                                                     # the served members, exactly as backtesting scores them
        _, o, idx = cv_evaluate(m, get_params(m, asset), d, n_splits=CV_FOLDS, return_oof=True)
        oof[m] = d['inv'](o)
    vs = {}
    for m in TABULAR:
        p, i = vol_scaled_oof(m, get_params(m, asset), d)
        assert np.array_equal(i, idx)
        vs[m] = p
    hub = {}
    for m, extra in HUBER.items():
        _, o, i = cv_evaluate(m, {**get_params(m, asset), **extra}, d, n_splits=CV_FOLDS, return_oof=True)
        assert np.array_equal(i, idx)
        hub[m] = d['inv'](o)
    y, zero = y_all[idx], np.zeros(len(idx))
    served = np.mean([oof[m] for m in ML], axis=0)
    cands = {'served: Combined (6 models)': served,
             'E6a: Combined, tabular members on the volatility-scaled target': np.mean([vs.get(m, oof[m]) for m in ML], axis=0),
             'E6b: Combined, LightGBM + CatBoost with Huber loss': np.mean([hub.get(m, oof[m]) for m in ML], axis=0)}
    rows = [{'asset': asset, **score(n, p, y, served, zero)} for n, p in cands.items()]
    for m in TABULAR:                                                # each member in isolation: candidate vs the same model as served
        rows.append({'asset': asset, **score(f'E6a detail: {m} vol-scaled vs served', vs[m], y, oof[m], zero)})
    for m in HUBER:
        rows.append({'asset': asset, **score(f'E6b detail: {m} Huber vs served', hub[m], y, oof[m], zero)})
    return rows


def main():
    rows = []
    for a in ASSETS:
        rows += run_asset(a)
        log.info(f"{a}: done")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EXPERIMENTS_DIR, 'E6_final_candidates.csv'), index=False)
    comb = df[~df['candidate'].str.contains('detail')]
    adopted = [c for c in comb['candidate'].unique() if not c.startswith('served') and
               all((r['RMSE_vs_served_pct'] < 0) and (r['DM_p_vs_served'] < 0.05) for _, r in comb[comb['candidate'] == c].iterrows())]
    lines = ["# E6 — submission-audit candidates (walk-forward folds only; the test set is never read)", "",
             "Decision rule (fixed before running): replace the served forecast only if the candidate lowers the pooled out-of-fold RMSE "
             "for every asset with DM p < 0.05 against it.", "",
             "| Asset | Candidate | RMSE vs random walk | RMSE vs served | DM p vs served | Dir. Acc % (p) | UP calls % |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in comb.iterrows():
        dms = '—' if pd.isna(r['DM_p_vs_served']) else f"{r['DM_p_vs_served']:.3f}"
        lines.append(f"| {r['asset']} | {r['candidate']} | {r['RMSE_vs_naive_pct']:+.3f} % | {r['RMSE_vs_served_pct']:+.3f} % | {dms} | "
                     f"{r['DirAcc_pct']:.1f} ({r['DirAcc_pvalue']:.3f}) | {r['UpCalls_pct']:.0f} |")
    lines += ["", "Each member in isolation (candidate vs the same model as served; negative = candidate better):", "",
              "| Asset | Member | RMSE change | DM p |", "|---|---|---:|---:|"]
    for _, r in df[df['candidate'].str.contains('detail')].iterrows():
        lines.append(f"| {r['asset']} | {r['candidate'].split(': ')[1]} | {r['RMSE_vs_served_pct']:+.3f} % | {r['DM_p_vs_served']:.3f} |")
    lines += ["", f"**Decision:** {'adopt ' + ', '.join(adopted) if adopted else 'no candidate meets the rule — the served forecast is unchanged.'}"]
    with open(os.path.join(EXPERIMENTS_DIR, 'E6_summary.md'), 'w') as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == '__main__':
    main()
