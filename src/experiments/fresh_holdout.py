"""
Fresh hold-out: the FROZEN models scored on days that are newer than the frozen dataset.

    python src/experiments/fresh_holdout.py            # uses the live download already in data/processed/
    python src/experiments/fresh_holdout.py --sync     # first downloads the latest complete bars (Yahoo Finance)

Why: the stored test window (2026-02-18 → 2026-09-11/12) was inspected during development (README, Limitations), so it is
not a pristine one-shot holdout. Every daily close published AFTER the frozen dataset was never seen by any version of
this project — not for training, tuning, model selection, the Combined rule or the band. Scoring the unchanged models on
those days is therefore the cleanest out-of-sample evidence available.

Nothing is refitted. For each new day t (the last observed day; t+1 = the day predicted) every trained model sees only the
30-day window ending on t, scaled with the train-fitted scaler — exactly the served inference path
(src/inference/prediction.py). The decision rule is fixed in advance:

    the Combined forecast shows skill on these days only if its RMSE (return) is below the random walk's
    AND the Diebold–Mariano test against the random walk gives p < 0.05.

Caveats printed with the result: the sample is small (≈ 40–60 days, so the tests have little power), and the live series
is a fresh Yahoo download (history can be revised slightly; GC=F / SI=F follow the front-month futures contract, so a
contract roll can create one artificial daily return). The overlap with the frozen data is checked and reported.

Outputs (the frozen results in results/*.csv are never touched):
    results/experiments/fresh_holdout.csv     one row per asset × forecast
    results/experiments/fresh_holdout.md      readable summary
"""
import os
import sys
import argparse
import datetime as dt
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

from config import ASSETS, ASSET_CONFIG, MODELS_DIR, PROCESSED_DATA_DIR, EXPERIMENTS_DIR, SEQ_LEN, VAL_END, get_prefix
from src.inference.prediction import load_features, combined_members, _load, _target_stats, COMBINED
from src.utils.metrics import evaluate_forecast

PPY = {a: (365 if ASSET_CONFIG[a]['type'] == 'crypto' else 252) for a in ASSETS}
RULE = 'skill only if Combined RMSE(return) < random walk AND Diebold–Mariano p < 0.05'


def fresh_days(asset):
    """Score every model on the days after the frozen dataset. Returns (rows, info) or (None, reason)."""
    live_path = os.path.join(PROCESSED_DATA_DIR, f'{get_prefix(asset)}_live_features.csv')
    if not os.path.exists(live_path):
        return None, 'no live download (run with --sync, or press "Sync live market data" in the dashboard)'
    frozen = load_features(asset, live=False)
    live = load_features(asset, live=True)
    frozen_end = frozen.index[-1]

    # data-revision check on the overlapping history (last 250 shared days)
    common = frozen.index.intersection(live.index)[-250:]
    revision = float(np.nanmax(np.abs(live.loc[common, 'price'] / frozen.loc[common, 'price'] - 1)) * 100) if len(common) else np.nan

    pos = np.where(live.index >= frozen_end)[0]                 # day t from the frozen end on: t+1 is newer than the frozen data
    pos = pos[pos + 1 < len(live)]                              # t+1 must exist
    if len(pos) < 10:
        return None, f'only {len(pos)} new days after {frozen_end.date()} — too few to score'

    scaler = joblib.load(os.path.join(MODELS_DIR, f'{get_prefix(asset)}_scaler.pkl'))
    cols = list(scaler.feature_names_in_)
    scaled = scaler.transform(live[cols].values)
    Xseq = np.stack([scaled[p - SEQ_LEN + 1:p + 1] for p in pos])
    Xt = scaled[pos]
    prev, nxt = live['price'].values[pos], live['price'].values[pos + 1]
    actual = np.log(nxt / prev)
    mu, sd = _target_stats(asset)

    preds = {m: np.ravel(_load(asset, m).predict(Xseq, Xt)) * sd + mu for m in combined_members(asset)}
    preds[COMBINED] = np.mean(list(preds.values()), axis=0)
    # drift baseline: mean next-day return over train+val (what 'Naive-Mean' uses in the frozen evaluation)
    r_trval = frozen.loc[:pd.Timestamp(VAL_END), 'log_return'].values[1:]
    preds['Naive-Mean'] = np.full(len(actual), float(np.mean(r_trval)))
    preds['Naive'] = np.zeros(len(actual))

    rows = []
    for name, r in preds.items():
        met = evaluate_forecast(actual, r, prev, periods_per_year=PPY[asset])
        rows.append({'asset': asset, 'model': name, 'n_days': len(actual),
                     'first_target': str(live.index[pos[0] + 1].date()), 'last_target': str(live.index[pos[-1] + 1].date()),
                     'RMSE_ret': met['RMSE_ret'], 'RMSE_vs_naive_pct': (met['RMSE_ret'] / float(np.sqrt(np.mean(actual ** 2))) - 1) * 100,
                     'MAE_usd': met['MAE_usd'], 'MAPE_usd': met['MAPE_usd'], 'R2_ret': met['R2_ret'],
                     'DirAcc_pct': met['DirAcc_pct'] if name != 'Naive' else np.nan,
                     'DirAcc_pvalue': met['DirAcc_pvalue'] if name != 'Naive' else np.nan,
                     'UpCalls_pct': met['UpCalls_pct'], 'UpDays_pct': met['UpDays_pct'],
                     'DM_pvalue_vs_naive': met['DM_pvalue'] if name != 'Naive' else np.nan,
                     'max_revision_vs_frozen_pct': revision, 'frozen_end': str(frozen_end.date())})
    return rows, None


def write_markdown(df, skipped, path):
    now = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    lines = ['# Fresh hold-out — frozen models on days newer than the frozen dataset', '',
             f'Run {now}. Nothing was refitted; no version of the project had seen these days. Decision rule (fixed in advance): {RULE}.', '',
             '| Asset | Days (targets) | Forecast | RMSE vs random walk | DM p | Direction % (p) | UP calls % / up days % | MAPE % |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for _, r in df.iterrows():
        if r['model'] not in (COMBINED, 'Naive-Mean', 'Naive'):
            continue
        dm = '—' if pd.isna(r['DM_pvalue_vs_naive']) else f"{r['DM_pvalue_vs_naive']:.2f}"
        da = '—' if pd.isna(r['DirAcc_pct']) else f"{r['DirAcc_pct']:.1f} ({r['DirAcc_pvalue']:.2f})"
        lines.append(f"| {r['asset']} | {r['n_days']} ({r['first_target']} → {r['last_target']}) | {r['model']} | "
                     f"{r['RMSE_vs_naive_pct']:+.2f} % | {dm} | {da} | {r['UpCalls_pct']:.0f} / {r['UpDays_pct']:.0f} | {r['MAPE_usd']:.2f} |")
    lines += ['', '**Verdict per asset (rule above):**', '']
    for a, g in df.groupby('asset', sort=False):
        c = g[g['model'] == COMBINED].iloc[0]
        skill = c['RMSE_vs_naive_pct'] < 0 and c['DM_pvalue_vs_naive'] < 0.05
        lines.append(f"- {a}: {'skill shown' if skill else 'no skill shown'} "
                     f"(Combined {c['RMSE_vs_naive_pct']:+.2f} % vs random walk, DM p = {c['DM_pvalue_vs_naive']:.2f}, {c['n_days']} days; "
                     f"max revision of the live download vs the frozen closes on the shared history: {c['max_revision_vs_frozen_pct']:.2f} %).")
    for a, why in skipped:
        lines.append(f'- {a}: not scored — {why}.')
    lines += ['', 'Caveats: ≈ 40–60 days give the tests little power (a real but small edge would usually not reach p < 0.05). '
                  'The live series is a fresh Yahoo download: history can be revised, and GC=F / SI=F follow the front-month contract, '
                  'so a contract roll can add one artificial daily return. Every single model is in fresh_holdout.csv.']
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sync', action='store_true', help='download the latest complete bars first (needs internet)')
    args = ap.parse_args()
    if args.sync:
        from src.data.sync_live_data import update_live_data
        for a in ASSETS:
            print(f'{a}: live sync ok = {update_live_data(a, refresh_external=True)}')
    rows, skipped = [], []
    for a in ASSETS:
        r, why = fresh_days(a)
        if r is None:
            skipped.append((a, why)); print(f'{a}: skipped — {why}')
        else:
            rows += r
    if not rows:
        print('Nothing scored.'); return None
    df = pd.DataFrame(rows)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(EXPERIMENTS_DIR, 'fresh_holdout.csv'), index=False)
    write_markdown(df, skipped, os.path.join(EXPERIMENTS_DIR, 'fresh_holdout.md'))
    print(open(os.path.join(EXPERIMENTS_DIR, 'fresh_holdout.md')).read())
    return df


if __name__ == '__main__':
    main()
