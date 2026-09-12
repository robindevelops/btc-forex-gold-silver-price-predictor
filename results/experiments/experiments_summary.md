# Experiment summary (validation only — the test set is never used here)

## E1 — Training-data size (fixed validation window 2025-09-11 → 2026-02-17)

| Asset | Model | 3 y: val RMSE (ret) | full: val RMSE (ret) | change | 3 y DA | full DA |
|---|---|---:|---:|---:|---:|---:|
| Bitcoin | Naive | 0.02610 | 0.02610 | +0.00% | — | — |
| Bitcoin | Ridge | 0.02649 | 0.02605 | -1.67% | 47.5 | 50.6 |
| Bitcoin | LightGBM | 0.02646 | 0.02619 | -1.01% | 47.5 | 47.5 |
| Bitcoin | CatBoost | 0.02641 | 0.02613 | -1.07% | 47.5 | 47.5 |
| Gold | Naive | 0.02035 | 0.02035 | +0.00% | — | — |
| Gold | Ridge | 0.02018 | 0.02024 | +0.32% | 62.4 | 54.1 |
| Gold | LightGBM | 0.02023 | 0.02028 | +0.25% | 63.3 | 62.4 |
| Gold | CatBoost | 0.02017 | 0.02008 | -0.43% | 63.3 | 60.6 |
| Silver | Naive | 0.05536 | 0.05536 | +0.00% | — | — |
| Silver | Ridge | 0.05524 | 0.05538 | +0.24% | 51.4 | 62.4 |
| Silver | LightGBM | 0.05529 | 0.05602 | +1.33% | 62.4 | 47.7 |
| Silver | CatBoost | 0.05553 | 0.05537 | -0.29% | 49.5 | 57.8 |

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
| Gold | Ridge | (none — all features) | +0.00% | 54.7 |
| Gold | Ridge | return path | +0.03% | 54.1 |
| Gold | Ridge | volatility state | +0.01% | 54.8 |
| Gold | Ridge | momentum / trend | +0.02% | 55.9 |
| Gold | Ridge | macro | +0.01% | 54.8 |
| Gold | LightGBM | (none — all features) | +0.00% | 53.8 |
| Gold | LightGBM | return path | +0.45% | 54.0 |
| Gold | LightGBM | volatility state | -0.04% | 54.2 |
| Gold | LightGBM | momentum / trend | +0.01% | 54.3 |
| Gold | LightGBM | macro | +0.09% | 53.7 |
| Silver | Ridge | (none — all features) | +0.00% | 53.3 |
| Silver | Ridge | return path | -0.01% | 53.3 |
| Silver | Ridge | volatility state | -0.04% | 53.5 |
| Silver | Ridge | momentum / trend | +0.01% | 52.6 |
| Silver | Ridge | macro | +0.03% | 53.6 |
| Silver | LightGBM | (none — all features) | +0.00% | 53.2 |
| Silver | LightGBM | return path | -0.15% | 52.3 |
| Silver | LightGBM | volatility state | -0.04% | 51.7 |
| Silver | LightGBM | momentum / trend | -0.13% | 53.5 |
| Silver | LightGBM | macro | -0.04% | 53.3 |

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
| Gold | return_1d | Ridge | -0.20% | -0.003 | 54.7 |
| Gold | return_1d | LightGBM | -0.11% | -0.004 | 53.8 |
| Gold | return_1d | CatBoost | -0.13% | -0.004 | 56.1 |
| Gold | return_5d | Naive | +0.00% | -0.040 | 41.4 |
| Gold | return_5d | Naive-Mean | -0.70% | -0.027 | 58.6 |
| Gold | return_5d | Ridge | -0.80% | -0.025 | 58.3 |
| Gold | return_5d | LightGBM | -0.60% | -0.030 | 57.0 |
| Gold | return_5d | CatBoost | -0.78% | -0.027 | 58.9 |
| Silver | return_1d | Naive | +0.00% | -0.002 | 0.0 |
| Silver | return_1d | Naive-Mean | -0.01% | -0.002 | 53.5 |
| Silver | return_1d | Ridge | -0.01% | -0.002 | 53.3 |
| Silver | return_1d | LightGBM | +0.15% | -0.006 | 53.2 |
| Silver | return_1d | CatBoost | +0.22% | -0.008 | 53.3 |
| Silver | return_5d | Naive | +0.00% | -0.013 | 0.0 |
| Silver | return_5d | Naive-Mean | -0.09% | -0.012 | 55.5 |
| Silver | return_5d | Ridge | -0.04% | -0.012 | 54.8 |
| Silver | return_5d | LightGBM | -0.21% | -0.012 | 54.2 |
| Silver | return_5d | CatBoost | +0.49% | -0.022 | 54.5 |

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
| Gold | Ridge | 0.3575 | -3.0% | -0.379 | 0.549 | 70.4 |
| Gold | LightGBM | 0.3680 | -0.2% | -0.450 | 0.548 | 69.3 |
| Gold | CatBoost | 0.3523 | -4.4% | -0.354 | 0.483 | 69.7 |
| Silver | Naive | 0.3388 | +0.0% | -0.216 | 0.334 | 0.0 |
| Silver | EWMA | 0.3239 | -4.4% | -0.100 | 0.311 | 59.5 |
| Silver | Naive-Mean | 0.4466 | +31.8% | -0.747 | 1.342 | 66.2 |
| Silver | Ridge | 0.4067 | +20.1% | -0.557 | 0.964 | 64.4 |
| Silver | LightGBM | 0.4538 | +33.9% | -0.856 | 1.299 | 64.4 |
| Silver | CatBoost | 0.4018 | +18.6% | -0.550 | 0.917 | 65.9 |
