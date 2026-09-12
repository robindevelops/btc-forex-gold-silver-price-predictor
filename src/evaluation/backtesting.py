"""
FINAL EVALUATION on the untouched test set — run once, after tuning and training.

    python src/evaluation/backtesting.py

For every asset it
  1. builds the walk-forward (validation) comparison table for baselines + all models
     (`results/cv_results.csv`) — this is what model SELECTION is based on;
  2. loads the deployed models and predicts the TEST period once
     (`results/final_test_results.csv`, `results/predictions/<prefix>_test_predictions.csv`);
  3. writes `data/models/model_status.json` — the model served per asset is the ML model with
     the best mean walk-forward RMSE (return space). The test set never influences selection.
  4. renders `results/FINAL_RESULTS.md` for the report.

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

from config import (ASSETS, RESULTS_DIR, MODELS_DIR, TUNING_DIR, MODEL_STATUS_PATH, PREDICTION_HORIZON_DAYS,
                    TRAIN_END, VAL_END, get_params, get_prefix, CV_FOLDS)
from src.data.preprocessing import build_dataset, unscale_target
from src.models.registry import make_model, load_trained, TABULAR, RECURRENT
from src.models.ensemble_model import StackedModel, stack_path
from src.evaluation.cross_validation import cv_evaluate, summarise_folds
from src.utils.metrics import evaluate_forecast
from src.utils.logging_config import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)
ML_MODELS = list(TABULAR) + list(RECURRENT)
BASELINE_MODELS = ['Naive-Zero', 'Naive-Mean', 'ARIMA']
PRED_DIR = os.path.join(RESULTS_DIR, 'predictions')
os.makedirs(PRED_DIR, exist_ok=True)


def model_exists(name, asset):
    p = get_prefix(asset)
    return os.path.exists(os.path.join(MODELS_DIR, f'{p}_{name.lower()}.pkl')) or \
        os.path.exists(os.path.join(MODELS_DIR, f'{p}_{name.lower()}.keras')) or \
        os.path.exists(os.path.join(MODELS_DIR, f'{p}_catboost.cbm')) and name == 'CatBoost'


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
    return rows


# ------------------------------------------------------------------ test predictions
def test_predictions(asset, data):
    scaler, ti = data['scaler'], data['target_idx']
    Xs, Xt, y = data['X_test'], data['Xt_test'], data['y_test']
    y_trval = np.concatenate([data['y_train'], data['y_val']])
    preds = {}
    zero_scaled = float(-scaler.data_min_[ti] / (scaler.data_max_[ti] - scaler.data_min_[ti]))
    preds['Naive-Zero'] = np.full(len(y), zero_scaled)
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
    return preds, infos


def main():
    cv_rows, test_rows, status = [], [], {}
    for asset in ASSETS:
        log.info(f"===== {asset} =====")
        data = build_dataset(asset)
        scaler, ti = data['scaler'], data['target_idx']
        true_ret = unscale_target(data['y_test'], scaler, ti)
        prev = data['prev_test']

        cv_rows += cv_table(asset, data)
        preds, infos = test_predictions(asset, data)

        pred_df = pd.DataFrame({'date': data['dates_test'], 'prev_close': prev,
                                'actual_close': data['true_test'], 'actual_return': true_ret})
        for name, p in preds.items():
            r = unscale_target(p, scaler, ti)
            met = evaluate_forecast(true_ret, r, prev)
            test_rows.append({'asset': asset, 'model': name, 'n_test': len(true_ret), **met,
                              'fit_info': json.dumps(infos.get(name, {}))})
            pred_df[f'pred_return_{name}'] = r
            pred_df[f'pred_close_{name}'] = prev * np.exp(r)
            log.info(f"  {name:12s} RMSE_ret={met['RMSE_ret']:.5f} R2_ret={met['R2_ret']:+.3f} "
                     f"DA={met['DirAcc_pct']:.1f}% (p={met['DirAcc_pvalue']:.2f}) DM p={met['DM_pvalue']:.2f} "
                     f"RMSE$={met['RMSE_usd']:,.2f} strat={met['strategy_return_pct']:+.1f}% B&H={met['buy_hold_return_pct']:+.1f}%")
        pred_df.to_csv(os.path.join(PRED_DIR, f'{get_prefix(asset)}_test_predictions.csv'), index=False)

        # ---- model selection on VALIDATION (walk-forward) results only
        cv_df = pd.DataFrame([r for r in cv_rows if r['asset'] == asset])
        ml = cv_df[cv_df['model'].isin(ML_MODELS) & cv_df['model'].isin(preds.keys())]
        best = ml.sort_values('RMSE_ret_mean').iloc[0]
        naive_cv = cv_df[cv_df['model'] == 'Naive-Zero'].iloc[0]
        test_best = [r for r in test_rows if r['asset'] == asset and r['model'] == best['model']][0]
        test_naive = [r for r in test_rows if r['asset'] == asset and r['model'] == 'Naive-Zero'][0]
        status[asset] = {
            'primary_model': best['model'], 'status': 'active',
            'selection_rule': 'lowest mean walk-forward RMSE (return space) on train+val folds',
            'cv_rmse_ret': float(best['RMSE_ret_mean']), 'cv_rmse_ret_naive': float(naive_cv['RMSE_ret_mean']),
            'cv_dir_acc': float(best['DirAcc_pct_mean']),
            'test': {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in test_best.items()
                     if k not in ('asset', 'fit_info')},
            'test_naive': {k: test_naive[k] for k in ('RMSE_ret', 'RMSE_usd', 'MAE_usd', 'MAPE_usd')},
            'horizon_days': PREDICTION_HORIZON_DAYS, 'train_end': TRAIN_END, 'val_end': VAL_END,
            'test_period': [str(data['dates_test'][0].date()), str(data['dates_test'][-1].date())],
            'params': get_params(best['model'], asset), 'fit_info': infos.get(best['model'], {}),
            'features': data['columns'],
        }
        log.info(f"  → served model for {asset}: {best['model']} (CV RMSE_ret {best['RMSE_ret_mean']:.5f} vs naive {naive_cv['RMSE_ret_mean']:.5f})")

    pd.DataFrame(cv_rows).to_csv(os.path.join(RESULTS_DIR, 'cv_results.csv'), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(RESULTS_DIR, 'final_test_results.csv'), index=False)
    with open(MODEL_STATUS_PATH, 'w') as f:
        json.dump(status, f, indent=2, default=str)
    write_markdown(pd.DataFrame(cv_rows), pd.DataFrame(test_rows), status)
    log.info("Final evaluation complete → results/final_test_results.csv, results/cv_results.csv, results/FINAL_RESULTS.md")


def write_markdown(cv_df, test_df, status):
    lines = ["# Final Results (auto-generated by src/evaluation/backtesting.py)", "",
             f"Split: train ≤ {TRAIN_END}, validation ≤ {VAL_END}, test = remainder (touched once).", "",
             "Model selection uses walk-forward validation only. **Bold** = served model. "
             "DA = directional accuracy on non-flat days (p = one-sided binomial test vs 50%). "
             "DM p = Diebold–Mariano test vs the zero-return random-walk forecast (squared error, return space).", ""]
    for asset in ASSETS:
        s = status[asset]
        lines += [f"## {asset}", "",
                  f"Test period {s['test_period'][0]} → {s['test_period'][1]} ({s['test']['n_test']} days). "
                  f"Served model: **{s['primary_model']}** (selected by CV RMSE {s['cv_rmse_ret']:.5f} vs naive {s['cv_rmse_ret_naive']:.5f}).", "",
                  "### Walk-forward validation (train+val, 4 expanding folds)", "",
                  "| Model | RMSE (ret) | MAE (ret) | R² (ret) | Dir. Acc % | RMSE ($) |", "|---|---:|---:|---:|---:|---:|"]
        for _, r in cv_df[cv_df['asset'] == asset].sort_values('RMSE_ret_mean').iterrows():
            b = '**' if r['model'] == s['primary_model'] else ''
            lines.append(f"| {b}{r['model']}{b} | {r['RMSE_ret_mean']:.5f} ± {r['RMSE_ret_std']:.5f} | {r['MAE_ret_mean']:.5f} | "
                         f"{r['R2_ret_mean']:+.3f} | {r['DirAcc_pct_mean']:.1f} | {r['RMSE_usd_mean']:,.2f} |")
        lines += ["", "### Untouched test set", "",
                  "| Model | MAE ($) | RMSE ($) | MAPE % | RMSE (ret) | R² (ret) | Dir. Acc % (n, p) | DM p vs naive | Strategy % | Buy&Hold % |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in test_df[test_df['asset'] == asset].sort_values('RMSE_ret').iterrows():
            b = '**' if r['model'] == s['primary_model'] else ''
            da = '—' if r['model'] == 'Naive-Zero' else f"{r['DirAcc_pct']:.1f} ({int(r['DirAcc_n'])}, {r['DirAcc_pvalue']:.2f})"
            dm = '—' if r['model'] == 'Naive-Zero' else f"{r['DM_pvalue']:.2f}"
            lines.append(f"| {b}{r['model']}{b} | {r['MAE_usd']:,.2f} | {r['RMSE_usd']:,.2f} | {r['MAPE_usd']:.2f} | {r['RMSE_ret']:.5f} | "
                         f"{r['R2_ret']:+.3f} | {da} | {dm} | {r['strategy_return_pct']:+.1f} | {r['buy_hold_return_pct']:+.1f} |")
        lines.append("")
    with open(os.path.join(RESULTS_DIR, 'FINAL_RESULTS.md'), 'w') as f:
        f.write("\n".join(lines))


if __name__ == '__main__':
    main()
