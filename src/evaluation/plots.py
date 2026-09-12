"""
Report figures, generated from the CSVs written by training/evaluation.

    python src/evaluation/plots.py        →  results/figures/*.png

Figures:
  price_history.png                     three assets with train / val / test shading
  model_comparison_cv.png               walk-forward RMSE (return) + directional accuracy, all models, all assets
  model_comparison_test.png             the same on the untouched test set
  <prefix>_actual_vs_predicted.png      test period: returns (line + scatter) and price (with the naive shadow)
  <prefix>_residuals.png                residual distribution and residuals over time
  <prefix>_loss_curves.png              GRU / LSTM training vs validation loss
  <prefix>_feature_importance.png       served model's feature importance (if available)
  <prefix>_strategy.png                 cumulative long/flat strategy vs buy-and-hold on test
  overfitting_gap.png                   train vs validation RMSE per model
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import ASSETS, RESULTS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR, TRAIN_END, VAL_END, get_prefix, load_model_status

plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 160, 'font.size': 10, 'axes.grid': True,
                     'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False})
COLORS = {'Bitcoin': '#F7931A', 'Gold': '#C9A227', 'Silver': '#7F8C8D'}
MODEL_COLORS = {'Naive': '#9E9E9E', 'Naive-Mean': '#BDBDBD', 'ARIMA': '#FF9800', 'Ridge': '#4CAF50',
                'RandomForest': '#795548', 'LightGBM': '#2196F3', 'CatBoost': '#9C27B0', 'GRU': '#00BCD4',
                'LSTM': '#3F51B5', 'Stacked': '#E91E63'}
PRED_DIR = os.path.join(RESULTS_DIR, 'predictions')


def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


def shade_splits(ax, index):
    ax.axvspan(index.min(), pd.Timestamp(TRAIN_END), color='#4CAF50', alpha=0.06, label='train')
    ax.axvspan(pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END), color='#FFC107', alpha=0.10, label='validation')
    ax.axvspan(pd.Timestamp(VAL_END), index.max(), color='#F44336', alpha=0.08, label='test')


def price_history():
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    for ax, asset in zip(axes, ASSETS):
        df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{get_prefix(asset)}_features.csv'), index_col=0, parse_dates=True)
        ax.plot(df.index, df['price'], color=COLORS[asset], lw=1.4)
        shade_splits(ax, df.index)
        ax.set_title(f'{asset} — daily close (USD), {len(df)} trading days')
        ax.set_ylabel('USD')
    axes[0].legend(loc='upper left', ncol=3)
    save(fig, 'price_history.png')


def comparison(kind):
    if kind == 'cv':
        df = pd.read_csv(os.path.join(RESULTS_DIR, 'cv_results.csv'))
        rmse_col, da_col, title = 'RMSE_ret_mean', 'DirAcc_pct_mean', 'Walk-forward validation (4 expanding folds, train+val)'
        err_col = 'RMSE_ret_std'
    else:
        df = pd.read_csv(os.path.join(RESULTS_DIR, 'final_test_results.csv'))
        rmse_col, da_col, title, err_col = 'RMSE_ret', 'DirAcc_pct', 'Untouched test set', None
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, asset in enumerate(ASSETS):
        a = df[df['asset'] == asset].sort_values(rmse_col, ascending=False)
        colors = [MODEL_COLORS.get(m, '#607D8B') for m in a['model']]
        ax = axes[0, j]
        ax.barh(a['model'], a[rmse_col], color=colors, xerr=a[err_col] if err_col else None, capsize=3)
        naive = a.loc[a['model'] == 'Naive', rmse_col]
        if len(naive):
            ax.axvline(float(naive.iloc[0]), color='red', ls='--', lw=1, label='random walk (naive)')
        ax.set_title(f'{asset}: RMSE of next-day log return (lower = better)')
        ax.set_xlim(a[rmse_col].min() * 0.9, a[rmse_col].max() * 1.05)
        ax.legend(loc='lower right', fontsize=8)
        ax = axes[1, j]
        b = a[a['model'] != 'Naive'].sort_values(da_col)
        ax.barh(b['model'], b[da_col], color=[MODEL_COLORS.get(m, '#607D8B') for m in b['model']])
        ax.axvline(50, color='red', ls='--', lw=1, label='coin flip (50%)')
        ax.set_xlim(30, 70)
        ax.set_title(f'{asset}: directional accuracy % (non-flat days)')
        ax.legend(loc='lower right', fontsize=8)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    save(fig, f'model_comparison_{kind}.png')


def per_asset(asset, status):
    prefix = get_prefix(asset)
    pred = pd.read_csv(os.path.join(PRED_DIR, f'{prefix}_test_predictions.csv'), parse_dates=['date', 'target_date'])
    pred['date'] = pred['target_date']   # plot against the day being predicted
    served = status[asset]['primary_model']
    test = pd.read_csv(os.path.join(RESULTS_DIR, 'final_test_results.csv'))
    met = test[(test['asset'] == asset) & (test['model'] == served)].iloc[0]
    r_true, r_pred = pred['actual_return'].values, pred[f'pred_return_{served}'].values

    # --- actual vs predicted: returns and price
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), gridspec_kw={'height_ratios': [1.2, 1, 1.2]})
    ax = axes[0]
    ax.plot(pred['date'], r_true * 100, color='black', lw=1, label='actual next-day return')
    ax.plot(pred['date'], r_pred * 100, color=MODEL_COLORS.get(served, 'C0'), lw=1.4, label=f'{served} prediction')
    ax.axhline(0, color='grey', lw=0.8)
    ax.set_ylabel('log return (%)')
    ax.set_title(f'{asset} — test period {pred["date"].min().date()} → {pred["date"].max().date()} ({len(pred)} days) — '
                 f'{served}: RMSE(ret)={met["RMSE_ret"]:.4f}, R²(ret)={met["R2_ret"]:+.3f}, DA={met["DirAcc_pct"]:.1f}%')
    ax.legend(loc='upper left', ncol=2)
    ax = axes[1]
    ax.scatter(r_true * 100, r_pred * 100, s=14, alpha=0.6, color=MODEL_COLORS.get(served, 'C0'))
    lim = np.abs(r_true * 100).max() * 1.05
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=0.8, label='perfect prediction')
    ax.axhline(0, color='grey', lw=0.6); ax.axvline(0, color='grey', lw=0.6)
    ax.set_xlabel('actual return (%)'); ax.set_ylabel('predicted return (%)')
    ax.set_xlim(-lim, lim)
    ax.set_title('Predicted vs actual returns (quadrants I/III = correct direction)')
    ax.legend(loc='upper left')
    ax = axes[2]
    ax.plot(pred['date'], pred['actual_close'], color='black', lw=1.2, label='actual close')
    ax.plot(pred['date'], pred[f'pred_close_{served}'], color=MODEL_COLORS.get(served, 'C0'), lw=1, ls='--', label=f'{served} predicted close')
    ax.plot(pred['date'], pred['pred_close_Naive'], color='#9E9E9E', lw=0.9, ls=':', label='naive (yesterday\'s close)')
    ax.set_ylabel('USD')
    naive_usd = test[(test['asset'] == asset) & (test['model'] == 'Naive')].iloc[0]['RMSE_usd']
    ax.set_title(f'Price view (RMSE \\${met["RMSE_usd"]:,.2f} vs naive \\${naive_usd:,.2f}) — a price line always tracks the actual with a 1-day lag; '
                 f'skill is only visible in the return plots above', fontsize=9)
    ax.legend(loc='upper left', ncol=3)
    save(fig, f'{prefix}_actual_vs_predicted.png')

    # --- residuals
    res = r_true - r_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(res * 100, bins=30, color=MODEL_COLORS.get(served, 'C0'), alpha=0.8, edgecolor='white')
    axes[0].axvline(0, color='red', ls='--')
    axes[0].set_title(f'{asset} {served} — residual distribution (mean {res.mean()*100:+.3f}%, std {res.std()*100:.2f}%)')
    axes[0].set_xlabel('actual − predicted return (%)')
    axes[1].plot(pred['date'], res * 100, color=MODEL_COLORS.get(served, 'C0'), lw=0.9)
    axes[1].axhline(0, color='red', ls='--')
    axes[1].set_title('Residuals over the test period (no trend ⇒ no systematic bias)')
    axes[1].set_ylabel('%')
    save(fig, f'{prefix}_residuals.png')

    # --- loss curves
    hist_dir = os.path.join(RESULTS_DIR, 'histories')
    curves = [(m, os.path.join(hist_dir, f'{asset.lower()}_{m.lower()}.json')) for m in ('GRU', 'LSTM')]
    curves = [(m, p) for m, p in curves if os.path.exists(p)]
    if curves:
        fig, axes = plt.subplots(1, len(curves), figsize=(6 * len(curves), 4), squeeze=False)
        for ax, (m, p) in zip(axes[0], curves):
            h = json.load(open(p))
            ax.plot(h['loss'], label='train loss (MSE, scaled)')
            ax.plot(h['val_loss'], label='validation loss')
            ax.axvline(int(np.argmin(h['val_loss'])), color='red', ls='--', lw=0.8, label='best epoch (restored)')
            ax.set_title(f'{asset} {m} — training vs validation loss'); ax.set_xlabel('epoch'); ax.legend()
        save(fig, f'{prefix}_loss_curves.png')

    # --- feature importance
    fi_path = os.path.join(RESULTS_DIR, 'feature_importance.csv')
    if os.path.exists(fi_path):
        fi = pd.read_csv(fi_path)
        fi = fi[(fi['asset'] == asset) & (fi['model'] == served)]
        if fi.empty:
            fi = pd.read_csv(fi_path); fi = fi[(fi['asset'] == asset) & (fi['model'] == 'LightGBM')]
        if not fi.empty:
            fi = fi.sort_values('importance').tail(20)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(fi['feature'], fi['importance'], color=MODEL_COLORS.get(fi['model'].iloc[0], 'C0'))
            ax.set_title(f'{asset} — {fi["model"].iloc[0]} feature importance (normalised gain)')
            save(fig, f'{prefix}_feature_importance.png')

    # --- strategy
    pos = (r_pred > 0).astype(float)
    trades = np.abs(np.diff(np.concatenate([[0.0], pos])))
    strat = np.cumsum(pos * r_true - trades * 10 / 1e4)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(pred['date'], (np.exp(np.cumsum(r_true)) - 1) * 100, color='black', lw=1.2, label='buy & hold')
    ax.plot(pred['date'], (np.exp(strat) - 1) * 100, color=MODEL_COLORS.get(served, 'C0'), lw=1.4,
            label=f'{served} long/flat strategy (10 bps cost, {int(trades.sum())} trades)')
    ax.axhline(0, color='grey', lw=0.8)
    ax.set_ylabel('cumulative return (%)')
    ax.set_title(f'{asset} — strategy backtest on the test set: strategy {met["strategy_return_pct"]:+.1f}% vs buy&hold {met["buy_hold_return_pct"]:+.1f}% '
                 f'(Sharpe {met["strategy_sharpe"]:.2f} vs {met["buy_hold_sharpe"]:.2f})')
    ax.legend(loc='upper left')
    save(fig, f'{prefix}_strategy.png')


def overfitting_gap():
    p = os.path.join(RESULTS_DIR, 'train_val_metrics.csv')
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, asset in zip(axes, ASSETS):
        a = df[df['asset'] == asset].pivot(index='model', columns='split', values='RMSE_ret')
        a = a.sort_values('val')
        x = np.arange(len(a))
        ax.bar(x - 0.2, a['train'], 0.4, label='train (in-sample)', color='#90CAF9')
        ax.bar(x + 0.2, a['val'], 0.4, label='validation (out-of-sample)', color='#1565C0')
        ax.set_xticks(x); ax.set_xticklabels(a.index, rotation=30, ha='right')
        # random-walk reference: RMSE of a zero forecast = RMS of the true returns in each split
        from src.data.preprocessing import build_dataset
        d = build_dataset(asset)
        for split, ls in (('train', ':'), ('val', '--')):
            rw = float(np.sqrt(np.mean(d[f'y_real_{split}'] ** 2)))
            ax.axhline(rw, color='red', ls=ls, lw=1, label=f'random walk ({split})')
        ax.set_title(f'{asset} — RMSE(return): train vs validation'); ax.legend(fontsize=7)
    fig.suptitle('Over-fitting check — compare each model\'s gap with the random-walk gap: the validation period is simply more volatile than training', fontweight='bold', fontsize=10)
    save(fig, 'overfitting_gap.png')


if __name__ == '__main__':
    status = load_model_status()
    print("Generating figures...")
    price_history()
    for kind in ('cv', 'test'):
        if os.path.exists(os.path.join(RESULTS_DIR, 'cv_results.csv' if kind == 'cv' else 'final_test_results.csv')):
            comparison(kind)
    overfitting_gap()
    for asset in ASSETS:
        if os.path.exists(os.path.join(PRED_DIR, f'{get_prefix(asset)}_test_predictions.csv')):
            per_asset(asset, status)
    print("Done.")
