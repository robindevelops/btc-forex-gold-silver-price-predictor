# Experiment summary (validation only — the test set is never used here)

## E1 — Training-data size (fixed validation window 2025-09-11 → 2026-02-17)

| Asset | Model | 3 y: val RMSE (ret) | full: val RMSE (ret) | change | 3 y DA | full DA |
|---|---|---:|---:|---:|---:|---:|
| Bitcoin | Naive | 0.02610 | 0.02610 | +0.00% | — | — |
| Bitcoin | Ridge | 0.02649 | 0.02605 | -1.67% | 47.5 | 50.6 |
| Bitcoin | LightGBM | 0.02646 | 0.02619 | -1.01% | 47.5 | 47.5 |
| Bitcoin | CatBoost | 0.02641 | 0.02613 | -1.07% | 47.5 | 47.5 |
| Gold | Naive | 0.02035 | 0.02035 | +0.00% | — | — |
| Gold | Ridge | 0.02017 | 0.02025 | +0.37% | 63.3 | 56.9 |
| Gold | LightGBM | 0.02023 | 0.02028 | +0.28% | 63.3 | 63.3 |
| Gold | CatBoost | 0.02011 | 0.02007 | -0.18% | 61.5 | 60.6 |
| Silver | Naive | 0.05536 | 0.05536 | +0.00% | — | — |
| Silver | Ridge | 0.05526 | 0.05543 | +0.32% | 53.2 | 59.6 |
| Silver | LightGBM | 0.05542 | 0.05583 | +0.74% | 40.4 | 49.5 |
| Silver | CatBoost | 0.05611 | 0.05532 | -1.40% | 46.8 | 62.4 |

## E2 — Feature-group ablation (walk-forward CV; positive change = dropping the group hurts)

| Asset | Model | Dropped group | RMSE change | DA |
|---|---|---|---:|---:|
| Bitcoin | Ridge | (none — all features) | +0.00% | 49.4 |
| Bitcoin | Ridge | return path | +0.02% | 49.2 |
| Bitcoin | Ridge | volatility state | -0.03% | 49.1 |
| Bitcoin | Ridge | momentum / trend | +0.03% | 50.6 |
| Bitcoin | Ridge | macro | +0.01% | 49.3 |
| Bitcoin | Ridge | sentiment / calendar / volume | +0.01% | 48.4 |
| Bitcoin | LightGBM | (none — all features) | +0.00% | 49.8 |
| Bitcoin | LightGBM | return path | +0.02% | 49.6 |
| Bitcoin | LightGBM | volatility state | -0.03% | 49.9 |
| Bitcoin | LightGBM | momentum / trend | -0.02% | 49.7 |
| Bitcoin | LightGBM | macro | +0.02% | 50.0 |
| Bitcoin | LightGBM | sentiment / calendar / volume | +0.04% | 49.1 |
| Gold | Ridge | (none — all features) | +0.00% | 54.8 |
| Gold | Ridge | return path | +0.02% | 54.9 |
| Gold | Ridge | volatility state | +0.01% | 54.6 |
| Gold | Ridge | momentum / trend | +0.02% | 55.6 |
| Gold | Ridge | macro | -0.00% | 55.3 |
| Gold | LightGBM | (none — all features) | +0.00% | 54.4 |
| Gold | LightGBM | return path | -0.03% | 54.6 |
| Gold | LightGBM | volatility state | -0.39% | 56.0 |
| Gold | LightGBM | momentum / trend | -0.36% | 55.1 |
| Gold | LightGBM | macro | -0.13% | 54.0 |
| Silver | Ridge | (none — all features) | +0.00% | 53.3 |
| Silver | Ridge | return path | -0.02% | 53.0 |
| Silver | Ridge | volatility state | -0.06% | 54.1 |
| Silver | Ridge | momentum / trend | +0.01% | 53.0 |
| Silver | Ridge | macro | -0.01% | 53.4 |
| Silver | LightGBM | (none — all features) | +0.00% | 53.0 |
| Silver | LightGBM | return path | -0.03% | 52.7 |
| Silver | LightGBM | volatility state | -0.10% | 52.7 |
| Silver | LightGBM | momentum / trend | -0.14% | 52.9 |
| Silver | LightGBM | macro | +0.05% | 53.9 |

## E3 — Horizon: next-day vs 5-day return (walk-forward CV)

| Asset | Task | Model | RMSE vs naive | R² (ret) | DA |
|---|---|---|---:|---:|---:|
| Bitcoin | return_1d | Naive | +0.00% | -0.005 | 0.0 |
| Bitcoin | return_1d | Naive-Mean | +0.11% | -0.007 | 48.8 |
| Bitcoin | return_1d | Ridge | +0.03% | -0.005 | 49.4 |
| Bitcoin | return_1d | LightGBM | +0.08% | -0.006 | 49.8 |
| Bitcoin | return_1d | CatBoost | -0.00% | -0.004 | 48.1 |
| Bitcoin | return_5d | Naive | +0.00% | -0.023 | 0.0 |
| Bitcoin | return_5d | Naive-Mean | +0.56% | -0.032 | 48.1 |
| Bitcoin | return_5d | Ridge | +0.40% | -0.030 | 47.8 |
| Bitcoin | return_5d | LightGBM | +0.53% | -0.033 | 49.1 |
| Bitcoin | return_5d | CatBoost | +1.40% | -0.048 | 50.8 |
| Gold | return_1d | Naive | +0.00% | -0.006 | 0.0 |
| Gold | return_1d | Naive-Mean | -0.12% | -0.004 | 55.8 |
| Gold | return_1d | Ridge | -0.18% | -0.003 | 54.8 |
| Gold | return_1d | LightGBM | +0.26% | -0.014 | 54.4 |
| Gold | return_1d | CatBoost | -0.03% | -0.006 | 55.8 |
| Gold | return_5d | Naive | +0.00% | -0.040 | 41.4 |
| Gold | return_5d | Naive-Mean | -0.70% | -0.027 | 58.6 |
| Gold | return_5d | Ridge | -0.81% | -0.025 | 57.9 |
| Gold | return_5d | LightGBM | -0.44% | -0.033 | 56.5 |
| Gold | return_5d | CatBoost | -1.04% | -0.022 | 60.0 |
| Silver | return_1d | Naive | +0.00% | -0.002 | 0.0 |
| Silver | return_1d | Naive-Mean | -0.01% | -0.002 | 53.5 |
| Silver | return_1d | Ridge | +0.05% | -0.003 | 53.3 |
| Silver | return_1d | LightGBM | +0.09% | -0.005 | 53.0 |
| Silver | return_1d | CatBoost | +0.01% | -0.002 | 52.5 |
| Silver | return_5d | Naive | +0.00% | -0.013 | 0.0 |
| Silver | return_5d | Naive-Mean | -0.09% | -0.012 | 55.5 |
| Silver | return_5d | Ridge | -0.05% | -0.012 | 54.7 |
| Silver | return_5d | LightGBM | +0.02% | -0.017 | 54.0 |
| Silver | return_5d | CatBoost | +0.46% | -0.022 | 54.5 |

## E4 — 22-day realised-volatility target (walk-forward CV)

| Asset | Model | RMSE (log vol) | vs persistence | R² (log vol) | QLIKE | DA (vol change) |
|---|---|---:|---:|---:|---:|---:|
| Bitcoin | Naive | 0.4452 | +0.0% | -0.646 | 0.588 | 0.0 |
| Bitcoin | EWMA | 0.4108 | -7.7% | -0.391 | 0.458 | 61.7 |
| Bitcoin | Naive-Mean | 0.4217 | -5.3% | -0.536 | 0.366 | 66.0 |
| Bitcoin | Ridge | 0.3938 | -11.5% | -0.327 | 0.344 | 66.2 |
| Bitcoin | LightGBM | 0.4058 | -8.8% | -0.410 | 0.418 | 64.9 |
| Bitcoin | CatBoost | 0.4062 | -8.8% | -0.399 | 0.442 | 65.7 |
| Gold | Naive | 0.3686 | +0.0% | -0.610 | 0.393 | 0.0 |
| Gold | EWMA | 0.3428 | -7.0% | -0.367 | 0.327 | 61.0 |
| Gold | Naive-Mean | 0.3863 | +4.8% | -0.563 | 0.682 | 69.4 |
| Gold | Ridge | 0.3577 | -2.9% | -0.381 | 0.549 | 70.3 |
| Gold | LightGBM | 0.3671 | -0.4% | -0.445 | 0.541 | 69.1 |
| Gold | CatBoost | 0.3510 | -4.8% | -0.347 | 0.472 | 70.0 |
| Silver | Naive | 0.3388 | +0.0% | -0.216 | 0.334 | 0.0 |
| Silver | EWMA | 0.3239 | -4.4% | -0.100 | 0.311 | 59.5 |
| Silver | Naive-Mean | 0.4466 | +31.8% | -0.747 | 1.342 | 66.2 |
| Silver | Ridge | 0.4071 | +20.2% | -0.560 | 0.967 | 64.4 |
| Silver | LightGBM | 0.4545 | +34.2% | -0.864 | 1.300 | 64.0 |
| Silver | CatBoost | 0.4012 | +18.4% | -0.553 | 0.899 | 64.9 |
