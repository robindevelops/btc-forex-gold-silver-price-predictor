"""
Exploratory data analysis (Chapter 3 / "Analysis" stage).

    python src/data/eda.py   →  results/eda_summary.csv, results/figures/eda_*.png

For each asset:
  * Augmented Dickey-Fuller test on the price level (expected: non-stationary) and on the
    log return (expected: stationary) — this is why the model target is the return.
  * ACF / PACF of log returns — shows how little linear autocorrelation daily returns have
    (why ARIMA and lag features can only help a little).
  * Return distribution: mean, std, skew, excess kurtosis (fat tails) and a histogram.
  * Volatility clustering: rolling 30-day σ.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR, ASSETS, TRAIN_END, get_prefix

COLORS = {'Bitcoin': '#F7931A', 'Gold': '#C9A227', 'Silver': '#7F8C8D'}


def adf(series):
    r = adfuller(series.dropna(), autolag='AIC')
    return {'adf_stat': r[0], 'p_value': r[1], 'stationary_5pct': r[1] < 0.05}


def main():
    rows = []
    fig_dist, ax_dist = plt.subplots(1, 3, figsize=(15, 4))
    fig_vol, ax_vol = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, asset in enumerate(ASSETS):
        df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{get_prefix(asset)}_features.csv'), index_col=0, parse_dates=True)
        train = df.loc[:TRAIN_END]                       # EDA statistics are computed on the training window only
        p, r = train['price'], train['log_return']
        a_p, a_r = adf(p), adf(r)
        ac = acf(r, nlags=10, fft=True)
        row = {'asset': asset, 'n_days_total': len(df), 'n_days_train': len(train),
               'start': df.index.min().date(), 'end': df.index.max().date(),
               'adf_price_p': a_p['p_value'], 'price_stationary': a_p['stationary_5pct'],
               'adf_return_p': a_r['p_value'], 'return_stationary': a_r['stationary_5pct'],
               'ret_mean_pct': r.mean() * 100, 'ret_std_pct': r.std() * 100, 'ret_skew': stats.skew(r),
               'ret_excess_kurtosis': stats.kurtosis(r), 'acf_lag1': ac[1], 'acf_lag2': ac[2], 'acf_lag5': ac[5],
               'share_abs_ret_gt_2pct': (r.abs() > 0.02).mean() * 100}
        rows.append(row)
        print(f"{asset:8s} ADF(price) p={a_p['p_value']:.3f}  ADF(return) p={a_r['p_value']:.2e}  σ={r.std()*100:.2f}%  "
              f"kurt={stats.kurtosis(r):.1f}  acf1={ac[1]:+.3f}")

        # ACF / PACF
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
        plot_acf(r, lags=30, ax=axes[0], alpha=0.05); axes[0].set_title(f'{asset} — ACF of daily log returns (train)')
        plot_pacf(r, lags=30, ax=axes[1], alpha=0.05, method='ywm'); axes[1].set_title('PACF')
        fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, f'eda_{get_prefix(asset)}_acf_pacf.png'), dpi=150); plt.close(fig)

        # distribution
        ax = ax_dist[i]
        ax.hist(r * 100, bins=50, density=True, color=COLORS[asset], alpha=0.7)
        x = np.linspace(r.min() * 100, r.max() * 100, 200)
        ax.plot(x, stats.norm.pdf(x, r.mean() * 100, r.std() * 100), 'k--', lw=1, label='normal fit')
        ax.set_title(f'{asset} returns: σ={r.std()*100:.2f}%, kurtosis={stats.kurtosis(r):.1f}')
        ax.set_xlabel('daily log return (%)'); ax.legend()

        # volatility clustering
        ax_vol[i].plot(df.index, df['log_return'].rolling(30).std() * 100, color=COLORS[asset])
        ax_vol[i].axvline(pd.Timestamp(TRAIN_END), color='red', ls='--', lw=0.8)
        ax_vol[i].set_title(f'{asset} — rolling 30-day volatility (%) — red line = end of training window')
        ax_vol[i].grid(alpha=0.3)

    fig_dist.suptitle('Return distributions are fat-tailed (kurtosis ≫ 0): large moves are far more common than a normal model expects', fontsize=10)
    fig_dist.tight_layout(); fig_dist.savefig(os.path.join(FIGURES_DIR, 'eda_return_distributions.png'), dpi=150)
    fig_vol.tight_layout(); fig_vol.savefig(os.path.join(FIGURES_DIR, 'eda_volatility_clustering.png'), dpi=150)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, 'eda_summary.csv'), index=False)
    print(f"Saved results/eda_summary.csv and figures/eda_*.png")


if __name__ == '__main__':
    main()
