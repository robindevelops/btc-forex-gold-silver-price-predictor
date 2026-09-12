"""
Before vs after comparison on IDENTICAL unseen days.

  before : the 3-year system (results/archive_3y_final/predictions/*.csv, evaluated 2026-02-18 → 2026-07-28)
  after  : the current system (results/predictions/*.csv), restricted to the same target dates

Writes results/experiments/before_after.csv and prints the comparison table.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import ASSETS, RESULTS_DIR, EXPERIMENTS_DIR, get_prefix, load_model_status
from src.utils.metrics import evaluate_forecast

OLD = os.path.join(RESULTS_DIR, 'archive_3y_final')


def metrics_from(pred_df, model, date_col):
    r = pred_df[f'pred_return_{model}'].values
    return evaluate_forecast(pred_df['actual_return'].values, r, pred_df['prev_close'].values)


def main():
    status_new = load_model_status()
    status_old = json.load(open(os.path.join(OLD, 'model_status_3y.json')))
    rows = []
    for asset in ASSETS:
        p = get_prefix(asset)
        old = pd.read_csv(os.path.join(OLD, 'predictions', f'{p}_test_predictions.csv'), parse_dates=['date'])
        new = pd.read_csv(os.path.join(RESULTS_DIR, 'predictions', f'{p}_test_predictions.csv'), parse_dates=['date', 'target_date'])
        common = sorted(set(old['date']) & set(new['target_date']))        # old 'date' == date being predicted
        o = old[old['date'].isin(common)].sort_values('date').reset_index(drop=True)
        n = new[new['target_date'].isin(common)].sort_values('target_date').reset_index(drop=True)
        # Yahoo revises a few historical closes between downloads (max 0.75 % on one BTC day, mean < 0.005 %);
        # each system is therefore also compared with the naive forecast computed on ITS OWN data.
        rev = float(np.abs(n['actual_close'].values / o['actual_close'].values - 1).max() * 100)
        m_old, m_new = status_old[asset]['primary_model'], status_new[asset]['primary_model']
        for label, df, model in (('naive (old data)', o, 'Naive'), ('before (3 y data)', o, m_old),
                                 ('naive (new data)', n, 'Naive'), ('after (full history)', n, m_new)):
            if label.startswith('naive'):
                met = evaluate_forecast(df['actual_return'].values, np.zeros(len(df)), df['prev_close'].values)
            else:
                met = metrics_from(df, model, 'date')
            rows.append({'asset': asset, 'system': label, 'model': model, 'n_days': len(df), 'max_price_revision_pct': rev,
                         'period': f"{common[0].date()} → {common[-1].date()}",
                         'MAE_usd': met['MAE_usd'], 'RMSE_usd': met['RMSE_usd'], 'MAPE_usd': met['MAPE_usd'],
                         'RMSE_ret': met['RMSE_ret'], 'R2_ret': met['R2_ret'], 'DirAcc_pct': met['DirAcc_pct'],
                         'DirAcc_pvalue': met['DirAcc_pvalue'], 'DM_pvalue_vs_naive': met['DM_pvalue']})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EXPERIMENTS_DIR, 'before_after.csv'), index=False)
    print(df.round(5).to_string(index=False))
    return df


if __name__ == '__main__':
    main()
