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
  4. calibrates the 68 % uncertainty band served with every forecast, P_t·exp(r̂ ± k·σ_t) with σ_t = EWMA volatility at
     day t: k is the 68.27 % quantile of |r − r̂|/σ_t over the walk-forward out-of-fold Combined errors whose target lies in
     the TRAINING split; the band is scored against the uncalibrated ±1σ band and a fixed-width band (calibrated the same
     way) on the validation split's out-of-fold days and once on the test set (`results/band_calibration.csv`,
     `model_status.json: band`);
  5. pools every walk-forward out-of-fold prediction (≈ 6 years, 1,584–2,328 days per asset, train+val only) and tests
     each model and the Combined forecast against the random walk on all of them at once — Diebold–Mariano and binomial
     direction tests with far more power than the single 143–207-day test window (`results/walkforward_pooled.csv`);
  6. renders `results/FINAL_RESULTS.md` for the report.

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
from src.utils.metrics import evaluate_task, band_calibration, diebold_mariano, directional_accuracy, rmse
from src.inference.prediction import BAND_SIGMA_FEATURE, BAND_RULE, BAND_NOMINAL
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
    comb_rows, oof, member_oof = combined_cv_row(asset, data)
    return rows + comb_rows, oof, member_oof


def combined_cv_row(asset, data):
    """
    Walk-forward score of the COMBINED forecast: the trained base models are refit per fold with their tuned
    parameters (exactly what the tuner did), their out-of-fold predictions are averaged with equal weights and
    the average is scored on each validation block. Only train+val is used.
    """
    members = [m for m in ML_MODELS if model_exists(m, asset)]
    if not members:
        return [], None, None
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
    return ([{'asset': asset, 'model': COMBINED, 'n_folds': len(folds), **summarise_folds(folds)}], (data['inv'](mean_pred), idx),
            ({m: data['inv'](oof[m]) for m in members}, idx))


def pooled_walkforward(asset, data, member_oof):
    """
    Every walk-forward out-of-fold prediction pooled into one long out-of-sample record (train+val only, ≈ 6 years) and
    tested against the random walk (r̂ = 0) and the drift forecast (always the training-window mean, i.e. 'always UP' for
    an asset that rose). Hyper-parameters were tuned on these folds, so any bias favours the models.
    """
    if member_oof is None:
        return []
    preds, idx = member_oof
    preds = dict(preds)
    preds[COMBINED] = np.mean(list(preds.values()), axis=0)
    _, nm_oof, nm_idx = cv_evaluate('Naive-Mean', {}, data, n_splits=CV_FOLDS, return_oof=True)
    assert np.array_equal(nm_idx, idx)
    preds['Naive-Mean'] = data['inv'](nm_oof)
    y = np.concatenate([data['y_real_train'], data['y_real_val']])[idx]
    dates = np.concatenate([data['dates_train'], data['dates_val']])[idx]
    zero = np.zeros(len(y))
    rows = []
    for name, p in preds.items():
        dm, p_dm = diebold_mariano(y, p, zero)
        dm_d, p_dm_d = diebold_mariano(y, p, preds['Naive-Mean'])
        da, n_da, p_da = directional_accuracy(y, p)
        rows.append({'asset': asset, 'model': name, 'n_days': len(y), 'first_day': str(pd.Timestamp(dates[0]).date()),
                     'last_day': str(pd.Timestamp(dates[-1]).date()), 'RMSE_ret': rmse(y, p), 'RMSE_ret_naive': rmse(y, zero),
                     'RMSE_vs_naive_pct': (rmse(y, p) / rmse(y, zero) - 1) * 100, 'DM_stat_vs_naive': dm, 'DM_pvalue': p_dm,
                     'DM_pvalue_vs_drift': (np.nan if name == 'Naive-Mean' else p_dm_d),
                     'DirAcc_pct': da, 'DirAcc_n': n_da, 'DirAcc_pvalue': p_da, 'UpCalls_pct': float(np.mean(p > 0) * 100),
                     'UpDays_pct': float(np.mean(y > 0) * 100), 'corr_pred_actual': float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 else np.nan})
    for r in rows:
        log.info(f"  pooled walk-forward {r['model']:12s} n={r['n_days']} RMSE vs naive {r['RMSE_vs_naive_pct']:+.3f}% "
                 f"DM p={r['DM_pvalue']:.3f} DA={r['DirAcc_pct']:.1f}% (p={r['DirAcc_pvalue']:.3f}) UP={r['UpCalls_pct']:.0f}%")
    return rows


def band_rows(asset, data, pred_df, oof):
    """
    Calibration of the served 68 % band, P_t·exp(r̂ ± k·σ_t), σ_t = `ewma_vol` at day t.

    k is fitted ONCE on the walk-forward out-of-fold Combined errors whose target lies in the TRAINING split:
        k = 68.27 % quantile of |r − r̂| / σ_t.
    Daily returns are fat-tailed, so the plain ±1σ band (k = 1) over-covers (~73–75 % instead of 68 %); k < 1 fixes that.
    Three bands are scored on the validation split's out-of-fold days (out-of-sample for k) and once on the test set:
        conditional         k·σ_t      (served)
        conditional_1sigma  σ_t        (the previous, uncalibrated band)
        fixed               w          constant half-width, the 68.27 % quantile of |r − r̂| on the same training errors
    Returns (rows, k).
    """
    f = data['features']
    if oof is None:
        raise ValueError(f"{asset}: no walk-forward predictions to calibrate the band")
    pred, idx = oof
    n_tr = len(data['y_train'])
    dates_all = np.concatenate([data['dates_train'], data['dates_val']])[idx]
    y_all = np.concatenate([data['y_real_train'], data['y_real_val']])[idx]
    sig_all = f[BAND_SIGMA_FEATURE].reindex(dates_all).values
    fit = idx < n_tr                                                 # out-of-fold days of the training split → calibration
    err = np.abs(y_all - pred)
    k = float(np.quantile(err[fit] / sig_all[fit], BAND_NOMINAL))
    w = float(np.quantile(err[fit], BAND_NOMINAL))
    log.info(f"  band calibration on {int(fit.sum())} training-split out-of-fold days: k = {k:.3f}, fixed half-width = {w * 100:.2f}%")
    rows = []
    va = ~fit                                                        # out-of-fold days of the validation split → out-of-sample check
    for name, s in (('conditional', k * sig_all[va]), ('conditional_1sigma', sig_all[va]), ('fixed', np.full(int(va.sum()), w))):
        rows.append({'asset': asset, 'split': 'validation', 'band': name, 'k': k, **band_calibration(y_all[va], pred[va], s)})
    y, r = pred_df['actual_return'].values, pred_df[f'pred_return_{COMBINED}'].values
    sig_c = f[BAND_SIGMA_FEATURE].reindex(pred_df['date']).values
    for name, s in (('conditional', k * sig_c), ('conditional_1sigma', sig_c), ('fixed', np.full(len(y), w))):
        rows.append({'asset': asset, 'split': 'test', 'band': name, 'k': k, **band_calibration(y, r, s)})
    pred_df['sigma_t'] = sig_c
    pred_df[f'band_lo_{COMBINED}'] = pred_df['prev_close'] * np.exp(r - k * sig_c)
    pred_df[f'band_hi_{COMBINED}'] = pred_df['prev_close'] * np.exp(r + k * sig_c)
    pred_df[f'in_band_{COMBINED}'] = ((pred_df['actual_close'] >= pred_df[f'band_lo_{COMBINED}']) &
                                      (pred_df['actual_close'] <= pred_df[f'band_hi_{COMBINED}'])).astype(int)
    for rr in rows:
        log.info(f"  band {rr['split']:10s} {rr['band']:18s} coverage={rr['coverage_pct']:.1f}% (nominal 68.3) mean half-width={rr['mean_sigma_pct']:.2f}%")
    return rows, k


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
    cv_rows, test_rows, status, regime_rows, band_tbl, pooled_rows = [], [], {}, [], [], []
    for asset in ASSETS:
        log.info(f"===== {asset} =====")
        data = build_dataset(asset)
        true_ret = data['y_real_test']
        prev = data['prev_test']

        rows_cv, oof, member_oof = cv_table(asset, data)
        cv_rows += rows_cv
        pooled = pooled_walkforward(asset, data, member_oof)
        pooled_rows += pooled
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
        b_rows, band_k = band_rows(asset, data, pred_df, oof) if COMBINED in preds else ([], None)
        band_tbl += b_rows
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
            'combined_walkforward_pooled': next((clean(r) for r in pooled if r['model'] == COMBINED), None),
            'band': {'rule': BAND_RULE, 'sigma_feature': BAND_SIGMA_FEATURE, 'nominal_coverage_pct': BAND_NOMINAL * 100, 'k': band_k,
                     **{f"{r['split']}_{r['band']}_coverage_pct": r['coverage_pct'] for r in b_rows},
                     **{f"{r['split']}_{r['band']}_qlike": r['qlike'] for r in b_rows}},
        }
        log.info(f"  → best single model for {asset} by CV: {best['model']} (CV RMSE_ret {best['RMSE_ret_mean']:.5f} vs naive "
                 f"{naive_cv['RMSE_ret_mean']:.5f}); served forecast = {COMBINED} of {status[asset]['combined_members']}")
        regime_rows += regime_analysis(asset, data, pred_df, COMBINED if COMBINED in preds else best['model'])

    pd.DataFrame(regime_rows).to_csv(os.path.join(RESULTS_DIR, 'regime_analysis.csv'), index=False)
    pd.DataFrame(band_tbl).to_csv(os.path.join(RESULTS_DIR, 'band_calibration.csv'), index=False)
    pd.DataFrame(pooled_rows).to_csv(os.path.join(RESULTS_DIR, 'walkforward_pooled.csv'), index=False)
    pd.DataFrame(cv_rows).to_csv(os.path.join(RESULTS_DIR, 'cv_results.csv'), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(RESULTS_DIR, 'final_test_results.csv'), index=False)
    with open(MODEL_STATUS_PATH, 'w') as f:
        json.dump(status, f, indent=2, default=str)
    write_markdown(pd.DataFrame(cv_rows), pd.DataFrame(test_rows), status, pd.DataFrame(band_tbl), pd.DataFrame(pooled_rows))
    log.info("Final evaluation complete → results/final_test_results.csv, results/cv_results.csv, results/band_calibration.csv, "
             "results/walkforward_pooled.csv, results/FINAL_RESULTS.md")


def write_markdown(cv_df, test_df, status, band_df=None, pooled_df=None):
    lines = ["# Final Results (auto-generated by src/evaluation/backtesting.py)", "",
             f"Split: train ≤ {TRAIN_END}, validation ≤ {VAL_END}, test = remainder (touched once).", "",
             "**Bold** = the served forecast (Combined: equal-weight mean of the trained base models, no fitted weights). "
             "*Italic* = the best single model by walk-forward validation (test set never used for selection). "
             "DA = directional accuracy on non-flat days (p = one-sided binomial test vs 50%); UP calls = share of days the model "
             "predicted a rise (compare with the share of actual up days in the section heading — a model that almost always says UP "
             "scores the asset's drift, not a signal). "
             "DM p = Diebold–Mariano test vs the zero-return random-walk forecast (squared error, return space).", ""]
    for asset in ASSETS:
        s = status[asset]
        lines += [f"## {asset}", "",
                  f"Test period {s['test_period'][0]} → {s['test_period'][1]} ({s['test']['n_test']} days, "
                  f"{s['combined_test'].get('UpDays_pct', float('nan')):.0f}% of them up days). "
                  f"Served forecast: **Combined** of {', '.join(s['combined_members'])}. "
                  f"Best single model by CV: *{s['primary_model']}* (CV RMSE {s['cv_rmse_ret']:.5f} vs naive {s['cv_rmse_ret_naive']:.5f}).", "",
                  "### Walk-forward validation (train+val, 4 expanding folds)", "",
                  "| Model | RMSE (ret) | MAE (ret) | R² (ret) | Dir. Acc % | RMSE ($) |", "|---|---:|---:|---:|---:|---:|"]
        for _, r in cv_df[cv_df['asset'] == asset].sort_values('RMSE_ret_mean').iterrows():
            b = '**' if r['model'] == COMBINED else ('*' if r['model'] == s['primary_model'] else '')
            lines.append(f"| {b}{r['model']}{b} | {r['RMSE_ret_mean']:.5f} ± {r['RMSE_ret_std']:.5f} | {r['MAE_ret_mean']:.5f} | "
                         f"{r['R2_ret_mean']:+.3f} | {r['DirAcc_pct_mean']:.1f} | {r['RMSE_usd_mean']:,.2f} |")
        lines += ["", "### Untouched test set", "",
                  "| Model | MAE ($) | RMSE ($) | MAPE % | RMSE (ret) | R² (ret) | Dir. Acc % (n, p) | UP calls % | DM p vs naive | Strategy % | Buy&Hold % |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in test_df[test_df['asset'] == asset].sort_values('RMSE_ret').iterrows():
            b = '**' if r['model'] == COMBINED else ('*' if r['model'] == s['primary_model'] else '')
            da = '—' if r['model'] == 'Naive' else f"{r['DirAcc_pct']:.1f} ({int(r['DirAcc_n'])}, {r['DirAcc_pvalue']:.2f})"
            dm = '—' if r['model'] == 'Naive' else f"{r['DM_pvalue']:.2f}"
            up = '—' if r['model'] == 'Naive' else f"{r['UpCalls_pct']:.0f}"
            lines.append(f"| {b}{r['model']}{b} | {r['MAE_usd']:,.2f} | {r['RMSE_usd']:,.2f} | {r['MAPE_usd']:.2f} | {r['RMSE_ret']:.5f} | "
                         f"{r['R2_ret']:+.3f} | {da} | {up} | {dm} | {r['strategy_return_pct']:+.1f} | {r['buy_hold_return_pct']:+.1f} |")
        if pooled_df is not None and len(pooled_df):
            pw = pooled_df[pooled_df['asset'] == asset].sort_values('RMSE_ret')
            p0 = pw.iloc[0]
            lines += ["", f"### Pooled walk-forward evidence ({int(p0['n_days'])} out-of-fold days, {p0['first_day']} → {p0['last_day']}, train+val only)", "",
                      "| Model | RMSE vs naive | DM p vs naive | DM p vs drift | Dir. Acc % (p) | UP calls % | corr(r̂, r) |", "|---|---:|---:|---:|---:|---:|---:|"]
            for _, r in pw.iterrows():
                b = '**' if r['model'] == COMBINED else ''
                dmd = '—' if pd.isna(r['DM_pvalue_vs_drift']) else f"{r['DM_pvalue_vs_drift']:.3f}"
                cr = '—' if pd.isna(r['corr_pred_actual']) else f"{r['corr_pred_actual']:+.3f}"
                lines.append(f"| {b}{r['model']}{b} | {r['RMSE_vs_naive_pct']:+.3f} % | {r['DM_pvalue']:.3f} | {dmd} | "
                             f"{r['DirAcc_pct']:.1f} ({r['DirAcc_pvalue']:.3f}) | {r['UpCalls_pct']:.0f} | {cr} |")
            lines.append(f"{p0['UpDays_pct']:.1f} % of these days were up days. Drift = Naive-Mean (the training-window mean return, "
                         "i.e. 'always UP' for an asset that rose); a directional hit-rate must beat the drift forecast's, not 50 %, to show timing skill. "
                         "Hyper-parameters were tuned on these folds, so the comparison is, if anything, biased in favour of the models.")
        if band_df is not None and len(band_df):
            kk = band_df[band_df['asset'] == asset]['k'].iloc[0]
            lines += ["", f"### 68 % uncertainty band served with the Combined forecast (P_t·exp(r̂ ± k·σ_t), k = {kk:.3f})", "",
                      "| Split | Band | Days | Coverage % (nominal 68.3) | Mean half-width % |", "|---|---|---:|---:|---:|"]
            for _, r in band_df[band_df['asset'] == asset].iterrows():
                b = '**' if r['band'] == 'conditional' else ''
                lines.append(f"| {r['split']} | {b}{r['band']}{b} | {int(r['n'])} | {r['coverage_pct']:.1f} | {r['mean_sigma_pct']:.2f} |")
            lines.append("conditional = k·σ_t with σ_t the EWMA volatility at day t (served); conditional_1sigma = the previous uncalibrated ±1σ_t band; "
                         "fixed = one constant half-width. k and the fixed width are calibrated on the walk-forward errors of the training split only, "
                         "so the validation and test rows are out-of-sample.")
        lines.append("")
    with open(os.path.join(RESULTS_DIR, 'FINAL_RESULTS.md'), 'w') as f:
        f.write("\n".join(lines))


if __name__ == '__main__':
    main()
