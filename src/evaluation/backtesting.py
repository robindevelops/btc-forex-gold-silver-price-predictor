"""
FINAL EVALUATION on the untouched test set — run once, after tuning and training.

    python src/evaluation/backtesting.py

For every asset it
  1. builds the walk-forward (validation) comparison table for baselines + all models
     (`results/cv_results.csv`) — this is what model SELECTION is based on;
  2. loads the deployed models and predicts the TEST period once
     (`results/final_test_results.csv`, `results/predictions/<prefix>_test_predictions.csv`);
  3. writes `data/models/model_status.json` — `primary_model` is the single ML model with the best mean
     walk-forward RMSE (return space); `combined_test` holds the test metrics of the served COMBINED forecast
     (equal-weight mean of every trained base model — no fitted weights, so nothing is selected on the test set).
  4. renders `results/FINAL_RESULTS.md` for the report.

The Combined forecast is scored twice, exactly like the single models: on the walk-forward folds (the base
models' out-of-fold predictions are averaged per fold) and once on the untouched test set.

Metrics (see src/utils/metrics.py): RMSE/MAE/R² in return space, RMSE/MAE/MAPE in USD,
directional accuracy on non-flat days with a binomial p-value, Diebold–Mariano test against
the zero-return (random walk) forecast, and a long/flat strategy backtest with 10 bps costs.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from config import (ASSETS, RESULTS_DIR, TUNING_DIR, MODEL_STATUS_PATH, PREDICTION_HORIZON_DAYS,
                    TRAIN_END, VAL_END, get_params, get_prefix, CV_FOLDS)
from src.data.preprocessing import build_dataset
from src.models.registry import make_model, load_trained, artefact_exists, TABULAR, RECURRENT
from src.models.ensemble_model import StackedModel, stack_path
from src.evaluation.cross_validation import cv_evaluate, summarise_folds
from src.utils.metrics import evaluate_task
from src.utils.logging_config import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)
ML_MODELS = list(TABULAR) + list(RECURRENT)
BASELINE_MODELS = ['Naive', 'Naive-Mean', 'ARIMA']
COMBINED = 'Combined'
PRED_DIR = os.path.join(RESULTS_DIR, 'predictions')
os.makedirs(PRED_DIR, exist_ok=True)


def model_exists(name, asset):
    return artefact_exists(name, asset)


# ------------------------------------------------------------------ validation table
def cv_table(asset, data):
    """Walk-forward scores for baselines (computed here) + tuned models (from tuning logs)."""
    rows = []
    for b in BASELINE_MODELS:
        folds = cv_evaluate(b, {}, data, n_splits=CV_FOLDS)
        rows.append({'asset': asset, 'model': b, 'n_folds': len(folds), **summarise_folds(folds)})
    summary_path = os.path.join(TUNING_DIR, 'tuning_summary.csv')
    if os.path.exists(summary_path):
        ts = pd.read_csv(summary_path)
        for _, r in ts[ts['asset'] == asset].iterrows():
            rows.append({'asset': asset, 'model': r['model'], 'n_folds': CV_FOLDS,
                         **{k: r[k] for k in ts.columns if k.endswith('_mean') or k.endswith('_std')}})
    rows += combined_cv_row(asset, data)
    return rows


def combined_cv_row(asset, data):
    """
    Walk-forward score of the COMBINED forecast: the trained base models are refit per fold with their tuned
    parameters (exactly what the tuner did), their out-of-fold predictions are averaged with equal weights and
    the average is scored on each validation block. Only train+val is used.
    """
    members = [m for m in ML_MODELS if model_exists(m, asset)]
    if not members:
        return []
    oof, fold_sizes, idx = {}, None, None
    for m in members:
        member_folds, oof[m], idx = cv_evaluate(m, get_params(m, asset, data['task']), data, n_splits=CV_FOLDS, return_oof=True)
        fold_sizes = [f['n_val'] for f in member_folds]          # identical for every member (same splits)
    mean_pred = np.mean([oof[m] for m in members], axis=0)
    y_real = np.concatenate([data['y_real_train'], data['y_real_val']])
    naive = np.concatenate([data['naive_train'], data['naive_val']])
    prev = np.concatenate([data['prev_train'], data['prev_val']])
    folds, start = [], 0
    for n_val in fold_sizes:
        va = idx[start:start + n_val]
        folds.append(evaluate_task(data, y_real[va], data['inv'](mean_pred[start:start + n_val]), naive[va], prev[va]))
        start += n_val
    log.info(f"{asset}: Combined walk-forward RMSE_ret={np.mean([f['RMSE_ret'] for f in folds]):.5f} over {len(folds)} folds "
             f"(members: {', '.join(members)})")
    return [{'asset': asset, 'model': COMBINED, 'n_folds': len(folds), **summarise_folds(folds)}]


# ------------------------------------------------------------------ test predictions
def test_predictions(asset, data):
    """Model-space predictions for the test split. Returns (preds, fit infos)."""
    Xs, Xt, y = data['X_test'], data['Xt_test'], data['y_test']
    y_trval = np.concatenate([data['y_train'], data['y_val']])
    preds = {}
    preds['Naive'] = data['fwd'](data['naive_test'])
    preds['Naive-Mean'] = np.full(len(y), float(y_trval.mean()))
    arima = make_model('ARIMA').fit_series(np.ravel(y_trval))
    preds['ARIMA'] = arima.forecast_walk_forward(np.ravel(y))
    log.info(f"{asset}: ARIMA order {arima.fit_info['order']}")
    infos = {'ARIMA': arima.fit_info}
    for name in ML_MODELS:
        if not model_exists(name, asset):
            log.warning(f"{asset}: no trained {name} found, skipping")
            continue
        m = load_trained(name, asset)
        preds[name] = np.ravel(m.predict(Xs, Xt))
        infos[name] = m.fit_info
    if os.path.exists(stack_path(asset)):
        st = StackedModel(asset)
        preds['Stacked'] = np.ravel(st.predict(Xs, Xt))
        infos['Stacked'] = st.fit_info
    # Combined = equal-weight mean of the base models' predictions (no fitted weights, so nothing is selected
    # on the test set). This is the forecast the dashboard and the API serve.
    members = [m for m in ML_MODELS if m in preds]
    if members:
        preds[COMBINED] = np.mean([preds[m] for m in members], axis=0)
        infos[COMBINED] = {'members': members, 'rule': 'equal-weight mean of predicted log returns'}
    return preds, infos


def regime_analysis(asset, data, pred_df, served):
    """Served forecast vs naive across market regimes of the unseen test period (regime defined on day t, i.e. before the prediction)."""
    f = data['features']
    d = pred_df.copy()
    vol = f['volatility_30d'].reindex(d['date']).values
    trend = f['return_20d'].reindex(d['date']).values
    q = np.nanquantile(vol, [1 / 3, 2 / 3])
    d['vol_regime'] = np.where(vol <= q[0], 'low volatility', np.where(vol <= q[1], 'medium volatility', 'high volatility'))
    d['trend_regime'] = np.where(trend > 0, 'up-trend (20d)', 'down-trend (20d)')
    d['day_regime'] = np.where(d['actual_return'] > 0, 'up day', 'down day')
    rows = []
    for col in ('vol_regime', 'trend_regime', 'day_regime'):
        for g, sub in d.groupby(col):
            rows.append({'asset': asset, 'regime_type': col, 'regime': g, 'n_days': len(sub),
                         'mae_pct_served': float(sub[f'error_pct_{served}'].abs().mean()),
                         'mae_pct_naive': float(sub['error_pct_Naive'].abs().mean()),
                         'dir_hit_served_pct': float(sub[f'direction_hit_{served}'].mean() * 100),
                         'rmse_ret_served': float(np.sqrt(np.mean((sub['actual_return'] - sub[f'pred_return_{served}']) ** 2))),
                         'rmse_ret_naive': float(np.sqrt(np.mean(sub['actual_return'] ** 2)))})
    return rows


def main():
    cv_rows, test_rows, status, regime_rows = [], [], {}, []
    for asset in ASSETS:
        log.info(f"===== {asset} =====")
        data = build_dataset(asset)
        true_ret = data['y_real_test']
        prev = data['prev_test']

        cv_rows += cv_table(asset, data)
        preds, infos = test_predictions(asset, data)

        # prediction history: one row per unseen day — what the model saw (date), what it predicted for
        # target_date, what actually happened, and the error
        pred_df = pd.DataFrame({'date': data['dates_test'], 'target_date': data['target_dates_test'], 'prev_close': prev,
                                'actual_close': data['true_test'], 'actual_return': true_ret})
        for name, p in preds.items():
            r = data['inv'](p)
            met = evaluate_task(data, true_ret, r, data['naive_test'], prev)
            test_rows.append({'asset': asset, 'model': name, 'n_test': len(true_ret), **met,
                              'fit_info': json.dumps(infos.get(name, {}))})
            pred_df[f'pred_return_{name}'] = r
            pred_df[f'pred_close_{name}'] = prev * np.exp(r)
            pred_df[f'error_pct_{name}'] = (prev * np.exp(r) - data['true_test']) / data['true_test'] * 100
            pred_df[f'direction_hit_{name}'] = (np.sign(r) == np.sign(true_ret)).astype(int)
            log.info(f"  {name:12s} RMSE_ret={met['RMSE_ret']:.5f} R2_ret={met['R2_ret']:+.3f} "
                     f"DA={met['DirAcc_pct']:.1f}% (p={met['DirAcc_pvalue']:.2f}) DM p={met['DM_pvalue']:.2f} "
                     f"RMSE$={met['RMSE_usd']:,.2f} strat={met['strategy_return_pct']:+.1f}% B&H={met['buy_hold_return_pct']:+.1f}%")
        pred_df.to_csv(os.path.join(PRED_DIR, f'{get_prefix(asset)}_test_predictions.csv'), index=False)

        # ---- model selection on VALIDATION (walk-forward) results only
        cv_df = pd.DataFrame([r for r in cv_rows if r['asset'] == asset])
        ml = cv_df[cv_df['model'].isin(ML_MODELS) & cv_df['model'].isin(preds.keys())]
        best = ml.sort_values('RMSE_ret_mean').iloc[0]
        naive_cv = cv_df[cv_df['model'] == 'Naive'].iloc[0]
        comb_cv = cv_df[cv_df['model'] == COMBINED]
        clean = lambda row: {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.items()
                             if k not in ('asset', 'fit_info')}
        test_best = [r for r in test_rows if r['asset'] == asset and r['model'] == best['model']][0]
        test_naive = [r for r in test_rows if r['asset'] == asset and r['model'] == 'Naive'][0]
        test_comb = ([r for r in test_rows if r['asset'] == asset and r['model'] == COMBINED] or [{}])[0]
        status[asset] = {
            'primary_model': best['model'], 'status': 'active',
            'selection_rule': 'lowest mean walk-forward RMSE (return space) on train+val folds',
            'cv_rmse_ret': float(best['RMSE_ret_mean']), 'cv_rmse_ret_naive': float(naive_cv['RMSE_ret_mean']),
            'cv_dir_acc': float(best['DirAcc_pct_mean']),
            'test': clean(test_best),
            'test_naive': {k: test_naive[k] for k in ('RMSE_ret', 'RMSE_usd', 'MAE_usd', 'MAPE_usd')},
            'served_forecast': COMBINED,
            'combined_members': infos.get(COMBINED, {}).get('members', []),
            'combined_rule': infos.get(COMBINED, {}).get('rule'),
            'combined_test': clean(test_comb),
            'combined_cv_rmse_ret': (float(comb_cv['RMSE_ret_mean'].iloc[0]) if len(comb_cv) else None),
            'combined_cv_dir_acc': (float(comb_cv['DirAcc_pct_mean'].iloc[0]) if len(comb_cv) else None),
            'horizon_days': PREDICTION_HORIZON_DAYS, 'train_end': TRAIN_END, 'val_end': VAL_END,
            'data_start': str(data['dates_train'][0].date()), 'n_train': int(len(data['y_train'])), 'n_val': int(len(data['y_val'])),
            'test_period': [str(data['target_dates_test'][0].date()), str(data['target_dates_test'][-1].date())],
            'params': get_params(best['model'], asset), 'fit_info': infos.get(best['model'], {}),
            'features': data['columns'], 'target_mean': data['target_mean'], 'target_std': data['target_std'],
        }
        log.info(f"  → best single model for {asset} by CV: {best['model']} (CV RMSE_ret {best['RMSE_ret_mean']:.5f} vs naive "
                 f"{naive_cv['RMSE_ret_mean']:.5f}); served forecast = {COMBINED} of {status[asset]['combined_members']}")
        regime_rows += regime_analysis(asset, data, pred_df, COMBINED if COMBINED in preds else best['model'])

    pd.DataFrame(regime_rows).to_csv(os.path.join(RESULTS_DIR, 'regime_analysis.csv'), index=False)
    pd.DataFrame(cv_rows).to_csv(os.path.join(RESULTS_DIR, 'cv_results.csv'), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(RESULTS_DIR, 'final_test_results.csv'), index=False)
    with open(MODEL_STATUS_PATH, 'w') as f:
        json.dump(status, f, indent=2, default=str)
    write_markdown(pd.DataFrame(cv_rows), pd.DataFrame(test_rows), status)
    log.info("Final evaluation complete → results/final_test_results.csv, results/cv_results.csv, results/FINAL_RESULTS.md")


def write_markdown(cv_df, test_df, status):
    lines = ["# Final Results (auto-generated by src/evaluation/backtesting.py)", "",
             f"Split: train ≤ {TRAIN_END}, validation ≤ {VAL_END}, test = remainder (touched once).", "",
             "**Bold** = the served forecast (Combined: equal-weight mean of the trained base models, no fitted weights). "
             "*Italic* = the best single model by walk-forward validation (test set never used for selection). "
             "DA = directional accuracy on non-flat days (p = one-sided binomial test vs 50%). "
             "DM p = Diebold–Mariano test vs the zero-return random-walk forecast (squared error, return space).", ""]
    for asset in ASSETS:
        s = status[asset]
        lines += [f"## {asset}", "",
                  f"Test period {s['test_period'][0]} → {s['test_period'][1]} ({s['test']['n_test']} days). "
                  f"Served forecast: **Combined** of {', '.join(s['combined_members'])}. "
                  f"Best single model by CV: *{s['primary_model']}* (CV RMSE {s['cv_rmse_ret']:.5f} vs naive {s['cv_rmse_ret_naive']:.5f}).", "",
                  "### Walk-forward validation (train+val, 4 expanding folds)", "",
                  "| Model | RMSE (ret) | MAE (ret) | R² (ret) | Dir. Acc % | RMSE ($) |", "|---|---:|---:|---:|---:|---:|"]
        for _, r in cv_df[cv_df['asset'] == asset].sort_values('RMSE_ret_mean').iterrows():
            b = '**' if r['model'] == COMBINED else ('*' if r['model'] == s['primary_model'] else '')
            lines.append(f"| {b}{r['model']}{b} | {r['RMSE_ret_mean']:.5f} ± {r['RMSE_ret_std']:.5f} | {r['MAE_ret_mean']:.5f} | "
                         f"{r['R2_ret_mean']:+.3f} | {r['DirAcc_pct_mean']:.1f} | {r['RMSE_usd_mean']:,.2f} |")
        lines += ["", "### Untouched test set", "",
                  "| Model | MAE ($) | RMSE ($) | MAPE % | RMSE (ret) | R² (ret) | Dir. Acc % (n, p) | DM p vs naive | Strategy % | Buy&Hold % |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in test_df[test_df['asset'] == asset].sort_values('RMSE_ret').iterrows():
            b = '**' if r['model'] == COMBINED else ('*' if r['model'] == s['primary_model'] else '')
            da = '—' if r['model'] == 'Naive' else f"{r['DirAcc_pct']:.1f} ({int(r['DirAcc_n'])}, {r['DirAcc_pvalue']:.2f})"
            dm = '—' if r['model'] == 'Naive' else f"{r['DM_pvalue']:.2f}"
            lines.append(f"| {b}{r['model']}{b} | {r['MAE_usd']:,.2f} | {r['RMSE_usd']:,.2f} | {r['MAPE_usd']:.2f} | {r['RMSE_ret']:.5f} | "
                         f"{r['R2_ret']:+.3f} | {da} | {dm} | {r['strategy_return_pct']:+.1f} | {r['buy_hold_return_pct']:+.1f} |")
        lines.append("")
    with open(os.path.join(RESULTS_DIR, 'FINAL_RESULTS.md'), 'w') as f:
        f.write("\n".join(lines))


if __name__ == '__main__':
    main()
