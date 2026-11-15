"""
Volatility forecast (#1, #3) and uncertainty-band volatility (#2) — evaluation and selection.
Implements results/volatility/PREREGISTRATION.md exactly; the decision rules are not tuned here.

    python src/volatility/evaluate.py                # both parts
    python src/volatility/evaluate.py --part vol     # #1 + #3 only
    python src/volatility/evaluate.py --part band    # #2 only (refits the six return models per fold: slow)

Selection uses ONLY the walk-forward out-of-fold days of train+val. The test split is read once at the end of each part,
after the selection is fixed, and its results are reported whatever they are.

Outputs: results/volatility/*.csv, results/volatility/VOLATILITY_RESULTS.md, data/models/<prefix>_vol_5d.json
(+ a tabular model artefact if one is selected) and data/models/<prefix>_band_vol.json.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from config import ASSETS, RESULTS_DIR, MODELS_DIR, CV_FOLDS, TRAIN_END, VAL_END, get_params, get_prefix
from src.data.preprocessing import build_dataset
from src.evaluation.cross_validation import cv_evaluate, walk_forward_splits, trainval_arrays
from src.models.registry import make_model, TABULAR, RECURRENT
from src.utils.metrics import rmse, qlike, diebold_mariano, band_calibration
from src.volatility.implied_vol import align_iv, IV_FOR_ASSET
from src.volatility.models import HARLog, HAR1d, GARCH11

OUT = os.path.join(RESULTS_DIR, 'volatility')
os.makedirs(OUT, exist_ok=True)
TASK = 'vol_5d'
H = 5
ML_VOL = ['Ridge', 'LightGBM', 'CatBoost']
BAND_NOMINAL = 0.6827
log = lambda *a: print(*a, flush=True)


def vol_scores(y, p, naive, h=H):
    """RMSE/QLIKE in log-vol space + DM vs a reference forecast (HAC with h−1 lags: 5-day targets overlap)."""
    ok = np.isfinite(p) & np.isfinite(y) & np.isfinite(naive)
    y, p, naive = y[ok], p[ok], naive[ok]
    dm, pv = diebold_mariano(y, p, naive, h=h)
    return {'n': int(ok.sum()), 'RMSE_log': rmse(y, p), 'RMSE_vs_ref_pct': (rmse(y, p) / rmse(y, naive) - 1) * 100,
            'QLIKE': qlike(np.exp(2 * y), np.exp(2 * p)), 'DM_stat': dm, 'DM_p': pv}


# ====================================================================== #1 + #3
def vol_part(asset):
    data = build_dataset(asset, task=TASK)
    feats = data['features']
    A = trainval_arrays(data)
    y = A['y_real']
    dates_tv = pd.DatetimeIndex(np.concatenate([data['dates_train'], data['dates_val']]))
    iv_tv = align_iv(asset, dates_tv).values
    splits = []
    for tr, va in walk_forward_splits(len(y), CV_FOLDS):
        splits.append((tr[:-(H - 1)], va))                    # purge exactly as cv_evaluate does for h > 1
    oof_idx = np.concatenate([va for _, va in splits])

    oof = {'Persistence': A['naive'][oof_idx],
           'EWMA': np.log(np.maximum(feats['ewma_vol'].reindex(dates_tv).values, 1e-6))[oof_idx]}
    for name, with_iv in (('HAR', False), ('HAR-IV', True)):
        preds = []
        for tr, va in splits:
            m = HARLog(with_iv).fit(feats, dates_tv[tr], y[tr], iv_tv[tr] if with_iv else None)
            preds.append(m.predict(feats, dates_tv[va], iv_tv[va] if with_iv else None))
        oof[name] = np.concatenate(preds)
    for name in ML_VOL:
        _, p, idx = cv_evaluate(name, get_params(name, asset, TASK), data, n_splits=CV_FOLDS, return_oof=True)
        assert np.array_equal(idx, oof_idx)
        oof[name] = data['inv'](p)
    y_oof = y[oof_idx]

    rows = []
    for name, p in oof.items():
        rows.append({'asset': asset, 'candidate': name, 'days': 'all out-of-fold', **vol_scores(y_oof, p, oof['Persistence'])})
    has_iv = np.isfinite(oof['HAR-IV'])
    for name, p in oof.items():
        rows.append({'asset': asset, 'candidate': name, 'days': f'out-of-fold days with {IV_FOR_ASSET[asset]}',
                     **vol_scores(y_oof[has_iv], p[has_iv], oof['Persistence'][has_iv])})

    # ---- decision (pre-registered)
    non_iv = [c for c in oof if c != 'HAR-IV']
    step1 = min(non_iv, key=lambda c: rmse(y_oof, oof[c]))
    choice, reason = step1, f'step 1: lowest pooled RMSE among non-IV candidates ({step1})'
    dm_iv, p_iv = diebold_mariano(y_oof[has_iv], oof['HAR-IV'][has_iv], oof[step1][has_iv], h=H)
    if rmse(y_oof[has_iv], oof['HAR-IV'][has_iv]) < rmse(y_oof[has_iv], oof[step1][has_iv]) and p_iv < 0.05:
        choice, reason = 'HAR-IV', reason + f'; step 2: HAR-IV beats it on {int(has_iv.sum())} IV days (DM p = {p_iv:.4f})'
    else:
        reason += f'; step 2: HAR-IV not adopted (DM p vs {step1} on IV days = {p_iv:.3f})'
    days = has_iv if choice == 'HAR-IV' else np.ones(len(y_oof), bool)
    _, p_vs_naive = diebold_mariano(y_oof[days], oof[choice][days], oof['Persistence'][days], h=H)
    served = bool(p_vs_naive < 0.05 and rmse(y_oof[days], oof[choice][days]) < rmse(y_oof[days], oof['Persistence'][days]))
    reason += f'; step 3: vs Persistence DM p = {p_vs_naive:.4g} → {"served" if served else "NOT served"}'
    log(f"{asset}: {reason}")

    # ---- final fit on train+val, then the ONE test evaluation
    iv_te = align_iv(asset, data['dates_test']).values
    y_te, naive_te = data['y_real_test'], data['naive_test']
    test_pred = {'Persistence': naive_te,
                 'EWMA': np.log(np.maximum(feats['ewma_vol'].reindex(data['dates_test']).values, 1e-6))}
    finals = {}
    for name, with_iv in (('HAR', False), ('HAR-IV', True)):
        finals[name] = HARLog(with_iv).fit(feats, dates_tv, y, iv_tv if with_iv else None)
        test_pred[name] = finals[name].predict(feats, data['dates_test'], iv_te if with_iv else None)
    for name in ML_VOL:
        m = make_model(name, get_params(name, asset, TASK))
        m.fit(data['X_train'], data['Xt_train'], data['y_train'], data['X_val'], data['Xt_val'], data['y_val'], final=True)
        finals[name] = m
        test_pred[name] = data['inv'](m.predict(data['X_test'], data['Xt_test']))
    test_rows = []
    for name, p in test_pred.items():
        test_rows.append({'asset': asset, 'candidate': name, 'selected': name == choice, 'served': served and name == choice,
                          **{f'{k}_vs_persistence' if k.startswith(('RMSE_vs', 'DM')) else k: v for k, v in vol_scores(y_te, p, naive_te).items()},
                          **{f'{k}_vs_ewma': v for k, v in vol_scores(y_te, p, test_pred['EWMA']).items() if k in ('RMSE_vs_ref_pct', 'DM_p')}})

    # ---- artefact
    vol_tr = np.exp(data['y_real_train'])
    art = {'asset': asset, 'task': TASK, 'horizon_days': H, 'target': 'ln sqrt(mean r² over the next 5 trading days)',
           'method': choice, 'served': served, 'decision': reason, 'iv_series': IV_FOR_ASSET[asset] if choice == 'HAR-IV' else None,
           'regime_thresholds_daily_vol': [float(np.quantile(vol_tr, 1 / 3)), float(np.quantile(vol_tr, 2 / 3))],
           'train_median_daily_vol': float(np.median(vol_tr)), 'fit_period': [str(dates_tv[0].date()), str(dates_tv[-1].date())],
           'walkforward': next(r for r in rows if r['candidate'] == choice and
                               r['days'] == ('all out-of-fold' if choice != 'HAR-IV' else f'out-of-fold days with {IV_FOR_ASSET[asset]}')),
           'test': next(r for r in test_rows if r['candidate'] == choice),
           'test_ewma': next(r for r in test_rows if r['candidate'] == 'EWMA'),
           'test_persistence': next(r for r in test_rows if r['candidate'] == 'Persistence')}
    if choice in ('HAR', 'HAR-IV'):
        art['model'] = finals[choice].to_dict()
    elif choice in ML_VOL:
        art['model'] = {'artefact': os.path.basename(finals[choice].save(asset, TASK)), 'name': choice,
                        'target_mean': data['target_mean'], 'target_std': data['target_std']}
    with open(os.path.join(MODELS_DIR, f'{get_prefix(asset)}_{TASK}.json'), 'w') as f:
        json.dump(art, f, indent=2, default=float)
    return rows, test_rows, art


# ====================================================================== #2
def combined_oof(asset, data):
    """Walk-forward out-of-fold Combined forecast (equal-weight mean of the six members, refit per fold), cached."""
    path = os.path.join(OUT, f'{get_prefix(asset)}_combined_oof.csv')
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=['date'])
    preds, idx = [], None
    for m in list(TABULAR) + list(RECURRENT):
        _, p, idx = cv_evaluate(m, get_params(m, asset), data, n_splits=CV_FOLDS, return_oof=True)
        preds.append(data['inv'](p))
        log(f"  {asset}: {m} out-of-fold done")
    A = trainval_arrays(data)
    dates = np.concatenate([data['dates_train'], data['dates_val']])[idx]
    df = pd.DataFrame({'date': dates, 'actual_return': A['y_real'][idx], 'pred_return': np.mean(preds, axis=0),
                       'split': np.where(idx < len(data['y_train']), 'train', 'val')})
    df.to_csv(path, index=False)
    return df


def band_part(asset):
    data = build_dataset(asset)
    feats = data['features']
    oof = combined_oof(asset, data)
    pooled = pd.read_csv(os.path.join(RESULTS_DIR, 'walkforward_pooled.csv'))
    ref = pooled[(pooled['asset'] == asset) & (pooled['model'] == 'Combined')]['RMSE_ret'].iloc[0]
    log(f"{asset}: recomputed Combined out-of-fold RMSE {rmse(oof['actual_return'], oof['pred_return']):.6f} vs stored {ref:.6f}")

    # σ candidates, parameters fitted on the TRAINING split only
    har = HAR1d().fit(feats, data['dates_train'], data['y_real_train'])
    r_all = feats['log_return']
    garch = GARCH11().fit(r_all.loc[:TRAIN_END].values)
    garch_sig = pd.Series(garch.sigma(r_all.values), index=r_all.index)
    sig = lambda name, dates: (feats['ewma_vol'].reindex(dates).values if name == 'EWMA' else
                               har.sigma(feats, dates) if name == 'HAR-1d' else garch_sig.reindex(dates).values)

    tr, va = oof[oof['split'] == 'train'], oof[oof['split'] == 'val']
    test = pd.read_csv(os.path.join(RESULTS_DIR, 'predictions', f'{get_prefix(asset)}_test_predictions.csv'), parse_dates=['date'])
    rows, ks = [], {}
    for name in ('EWMA', 'HAR-1d', 'GARCH(1,1)'):
        s_tr = sig(name, pd.DatetimeIndex(tr['date']))
        k = float(np.quantile(np.abs(tr['actual_return'] - tr['pred_return']) / s_tr, BAND_NOMINAL))
        ks[name] = k
        for split, d, rcol in (('validation', va, 'pred_return'), ('test', test, 'pred_return_Combined')):
            s = sig(name, pd.DatetimeIndex(d['date']))
            cal = band_calibration(d['actual_return'].values, d[rcol].values, k * s)
            q = qlike((d['actual_return'] - d[rcol]).values ** 2, s ** 2)       # σ itself as a variance forecast of the error
            rows.append({'asset': asset, 'sigma': name, 'split': split, 'k': k, 'n': cal['n'], 'coverage_pct': cal['coverage_pct'],
                         'coverage_error_pp': abs(cal['coverage_pct'] - BAND_NOMINAL * 100), 'qlike_sigma': q,
                         'mean_halfwidth_pct': cal['mean_sigma_pct']})
    v = {r['sigma']: r for r in rows if r['split'] == 'validation'}
    qualify = [c for c in ('HAR-1d', 'GARCH(1,1)')
               if v[c]['qlike_sigma'] < v['EWMA']['qlike_sigma'] and v[c]['coverage_error_pp'] <= v['EWMA']['coverage_error_pp']]
    choice = min(qualify, key=lambda c: v[c]['qlike_sigma']) if qualify else 'EWMA'
    reason = (f"{choice} replaces EWMA (validation QLIKE {v[choice]['qlike_sigma']:.4f} vs {v['EWMA']['qlike_sigma']:.4f}, "
              f"coverage {v[choice]['coverage_pct']:.1f}% vs {v['EWMA']['coverage_pct']:.1f}%)" if qualify else
              'no candidate beats EWMA on both validation QLIKE and coverage → EWMA stays: ' +
              '; '.join(f"{c} QLIKE {v[c]['qlike_sigma']:.4f}, coverage {v[c]['coverage_pct']:.1f}%" for c in v))
    log(f"{asset} band: {reason}")
    for r in rows:
        r['selected'] = r['sigma'] == choice
    art = {'asset': asset, 'sigma': choice, 'k': ks[choice], 'decision': reason,
           'har_1d': har.to_dict(), 'garch': garch.to_dict(), 'k_all': ks,
           'validation': v[choice], 'test': next(r for r in rows if r['split'] == 'test' and r['sigma'] == choice)}
    with open(os.path.join(MODELS_DIR, f'{get_prefix(asset)}_band_vol.json'), 'w') as f:
        json.dump(art, f, indent=2, default=float)
    return rows, art


def apply_band(assets):
    """
    Re-score the served band with the selected σ_t through the evaluation script's OWN band code (backtesting.band_rows),
    using the cached out-of-fold Combined predictions, and update what the dashboard reads: model_status.json['band'],
    results/band_calibration.csv and the band columns of results/predictions/*. Nothing else in those files changes.
    """
    from config import MODEL_STATUS_PATH, load_model_status
    from src.evaluation.backtesting import band_rows, PRED_DIR
    from src.volatility.band import band_rule, band_method
    status = load_model_status()
    bc_path = os.path.join(RESULTS_DIR, 'band_calibration.csv')
    bc = pd.read_csv(bc_path)
    for asset in assets:
        if band_method(asset) == 'EWMA':
            # unchanged method: keep the evaluation script's original numbers (a recomputation differs only by ~1e-11 because
            # the GRU/LSTM out-of-fold refits are not bit-exact) and only label the method
            status[asset].setdefault('band', {}).update({'rule': band_rule(asset), 'sigma_method': 'EWMA'})
            status[asset]['band'].pop('sigma_feature', None)
            log(f"{asset}: band σ = EWMA (unchanged); stored band results kept as they are")
            continue
        data = build_dataset(asset)
        oof = combined_oof(asset, data)
        tv_dates = pd.DatetimeIndex(np.concatenate([data['dates_train'], data['dates_val']]))
        idx = tv_dates.get_indexer(pd.DatetimeIndex(oof['date']))
        assert (idx >= 0).all()
        p_path = os.path.join(PRED_DIR, f'{get_prefix(asset)}_test_predictions.csv')
        raw = pd.read_csv(p_path, dtype=str)                               # untouched columns are written back byte-for-byte
        pred_df = pd.read_csv(p_path, parse_dates=['date', 'target_date'])
        rows, k = band_rows(asset, data, pred_df, (oof['pred_return'].values, idx))
        band_cols = ['sigma_t', 'band_lo_Combined', 'band_hi_Combined', 'in_band_Combined']
        before = raw[band_cols].copy()
        raw[band_cols] = pred_df[band_cols].astype(str).values
        if not np.allclose(before[band_cols[:3]].astype(float).values, pred_df[band_cols[:3]].values, rtol=1e-12):
            raw.to_csv(p_path, index=False)                                # only rewrite when the band actually changed
        bc = pd.concat([bc[bc['asset'] != asset], pd.DataFrame(rows)], ignore_index=True)
        status[asset]['band'] = {'rule': band_rule(asset), 'sigma_method': band_method(asset), 'nominal_coverage_pct': BAND_NOMINAL * 100,
                                 'k': k, **{f"{r['split']}_{r['band']}_coverage_pct": r['coverage_pct'] for r in rows},
                                 **{f"{r['split']}_{r['band']}_qlike": r['qlike'] for r in rows}}
        log(f"{asset}: served band σ = {band_method(asset)}, k = {k:.3f}, test coverage "
            f"{next(r['coverage_pct'] for r in rows if r['split'] == 'test' and r['band'] == 'conditional'):.1f}%")
    bc['_o'] = bc['asset'].map({a: i for i, a in enumerate(ASSETS)})
    bc.sort_values(['_o'], kind='stable').drop(columns='_o').to_csv(bc_path, index=False)
    with open(MODEL_STATUS_PATH, 'w') as f:
        json.dump(status, f, indent=2, default=str)


def write_report():
    """results/volatility/VOLATILITY_RESULTS.md from the CSV/JSON outputs (no computation)."""
    L = ["# Volatility forecast (#1, #3) and uncertainty band (#2) — results", "",
         "Auto-generated by `src/volatility/evaluate.py`. Decision rules: `PREREGISTRATION.md` (written before the run). "
         "Selection uses only walk-forward out-of-fold days of train+val; the test split was evaluated once afterwards.", "",
         "**Disclosed deviation.** The first run produced a degenerate HAR-IV fit for Bitcoin's first walk-forward fold, which ends in "
         "2019 — before DVOL exists — so the fit had 0 rows and predicted with all-zero coefficients. That bug was fixed (a fit needs "
         "≥ 250 complete rows, else no forecast) after the first run's test table had been printed. The decision rules were not changed, "
         "and the fix changed no decision (Bitcoin: Ridge before and after; HAR-IV vs Ridge DM p = 0.085 after the fix).", ""]
    wf, te = pd.read_csv(os.path.join(OUT, 'vol_walkforward.csv')), pd.read_csv(os.path.join(OUT, 'vol_test.csv'))
    dec = json.load(open(os.path.join(OUT, 'vol_decisions.json')))
    L += ["## #1 + #3 — risk forecast: volatility of the next 5 trading days", ""]
    for a in ASSETS:
        L += [f"### {a}", "", f"Decision: {dec[a]}", "", "| Candidate | Walk-forward days | RMSE (log vol) | vs persistence | DM p | Test RMSE | Test vs persistence (DM p) | Test vs EWMA (DM p) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        w = wf[(wf['asset'] == a) & (wf['days'] == 'all out-of-fold')].set_index('candidate')
        pv = lambda v, f='.3f': '—' if pd.isna(v) else format(v, f)
        for _, t in te[te['asset'] == a].iterrows():
            c = t['candidate']; b = '**' if t['selected'] else ''
            L.append(f"| {b}{c}{b} | {int(w.loc[c, 'n'])} | {w.loc[c, 'RMSE_log']:.4f} | {w.loc[c, 'RMSE_vs_ref_pct']:+.1f}% | "
                     f"{pv(w.loc[c, 'DM_p'], '.3g')} | {t['RMSE_log']:.4f} | "
                     f"{t['RMSE_vs_ref_pct_vs_persistence']:+.1f}% ({pv(t['DM_p_vs_persistence'])}) | "
                     f"{t['RMSE_vs_ref_pct_vs_ewma']:+.1f}% ({pv(t['DM_p_vs_ewma'])}) |")
        L.append("")
    L.append("Bold = selected on walk-forward (served). HAR-IV walk-forward rows cover only the days where the implied-volatility series exists "
             "(Bitcoin: DVOL from 2021-03; see vol_walkforward.csv for the like-for-like comparison on those days). Other test rows are shown "
             "for transparency and were not used for selection.")
    bpath = os.path.join(OUT, 'band_candidates.csv')
    if os.path.exists(bpath):
        bd, bdec = pd.read_csv(bpath), json.load(open(os.path.join(OUT, 'band_decisions.json')))
        L += ["", "## #2 — σ_t of the 68 % band", "", "| Asset | σ_t | k | Validation coverage | Validation QLIKE | Test coverage | Selected |", "|---|---|---:|---:|---:|---:|---|"]
        for _, r in bd.sort_values(['asset', 'sigma', 'split']).iterrows():
            if r['split'] != 'validation':
                continue
            t = bd[(bd['asset'] == r['asset']) & (bd['sigma'] == r['sigma']) & (bd['split'] == 'test')].iloc[0]
            L.append(f"| {r['asset']} | {r['sigma']} | {r['k']:.3f} | {r['coverage_pct']:.1f}% | {r['qlike_sigma']:.4f} | {t['coverage_pct']:.1f}% | "
                     f"{'**yes**' if r['selected'] else ''} |")
        L += ["", "Nominal coverage 68.3 %. Decisions:", ""] + [f"* {a}: {d}" for a, d in bdec.items()]
    with open(os.path.join(OUT, 'VOLATILITY_RESULTS.md'), 'w') as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['vol', 'band', 'both', 'apply-band', 'report'], default='both')
    ap.add_argument('--assets', nargs='+', default=ASSETS)
    args = ap.parse_args()
    if args.part in ('vol', 'both'):
        wf, te, arts = [], [], {}
        for a in args.assets:
            r, t, arts[a] = vol_part(a)
            wf += r; te += t
        pd.DataFrame(wf).to_csv(os.path.join(OUT, 'vol_walkforward.csv'), index=False)
        pd.DataFrame(te).to_csv(os.path.join(OUT, 'vol_test.csv'), index=False)
        with open(os.path.join(OUT, 'vol_decisions.json'), 'w') as f:
            json.dump({a: x['decision'] for a, x in arts.items()}, f, indent=2)
    if args.part in ('band', 'both'):
        rows, arts = [], {}
        for a in args.assets:
            r, arts[a] = band_part(a)
            rows += r
        pd.DataFrame(rows).to_csv(os.path.join(OUT, 'band_candidates.csv'), index=False)
        with open(os.path.join(OUT, 'band_decisions.json'), 'w') as f:
            json.dump({a: x['decision'] for a, x in arts.items()}, f, indent=2)
    if args.part in ('band', 'both', 'apply-band'):
        apply_band(args.assets)
    write_report()


if __name__ == '__main__':
    main()
