# E6 — submission-audit candidates (walk-forward folds only; the test set is never read)

Decision rule (fixed before running): replace the served forecast only if the candidate lowers the pooled out-of-fold RMSE for every asset with DM p < 0.05 against it.

| Asset | Candidate | RMSE vs random walk | RMSE vs served | DM p vs served | Dir. Acc % (p) | UP calls % |
|---|---|---:|---:|---:|---:|---:|
| Bitcoin | served: Combined (6 models) | -0.018 % | +0.000 % | — | 50.8 (0.234) | 67 |
| Bitcoin | E6a: Combined, tabular members on the volatility-scaled target | +0.147 % | +0.165 % | 0.016 | 48.4 (0.940) | 65 |
| Bitcoin | E6b: Combined, LightGBM + CatBoost with Huber loss | -0.009 % | +0.010 % | 0.794 | 50.1 (0.475) | 68 |
| Gold | served: Combined (6 models) | -0.213 % | +0.000 % | — | 52.5 (0.023) | 74 |
| Gold | E6a: Combined, tabular members on the volatility-scaled target | -0.182 % | +0.031 % | 0.700 | 52.7 (0.018) | 79 |
| Gold | E6b: Combined, LightGBM + CatBoost with Huber loss | -0.180 % | +0.034 % | 0.609 | 53.0 (0.010) | 78 |
| Silver | served: Combined (6 models) | -0.058 % | +0.000 % | — | 52.7 (0.016) | 78 |
| Silver | E6a: Combined, tabular members on the volatility-scaled target | -0.110 % | -0.053 % | 0.420 | 52.1 (0.046) | 80 |
| Silver | E6b: Combined, LightGBM + CatBoost with Huber loss | -0.043 % | +0.015 % | 0.545 | 52.2 (0.041) | 78 |

Each member in isolation (candidate vs the same model as served; negative = candidate better):

| Asset | Member | RMSE change | DM p |
|---|---|---:|---:|
| Bitcoin | Ridge vol-scaled vs served | +0.236 % | 0.063 |
| Bitcoin | RandomForest vol-scaled vs served | +0.379 % | 0.005 |
| Bitcoin | LightGBM vol-scaled vs served | +0.466 % | 0.032 |
| Bitcoin | CatBoost vol-scaled vs served | +0.321 % | 0.019 |
| Bitcoin | LightGBM Huber vs served | +0.211 % | 0.254 |
| Bitcoin | CatBoost Huber vs served | -0.210 % | 0.057 |
| Gold | Ridge vol-scaled vs served | +0.008 % | 0.915 |
| Gold | RandomForest vol-scaled vs served | -0.074 % | 0.656 |
| Gold | LightGBM vol-scaled vs served | -0.007 % | 0.935 |
| Gold | CatBoost vol-scaled vs served | +0.259 % | 0.486 |
| Gold | LightGBM Huber vs served | -0.134 % | 0.247 |
| Gold | CatBoost Huber vs served | +0.377 % | 0.345 |
| Silver | Ridge vol-scaled vs served | +0.197 % | 0.303 |
| Silver | RandomForest vol-scaled vs served | -0.327 % | 0.286 |
| Silver | LightGBM vol-scaled vs served | -0.033 % | 0.718 |
| Silver | CatBoost vol-scaled vs served | -0.035 % | 0.468 |
| Silver | LightGBM Huber vs served | +0.022 % | 0.797 |
| Silver | CatBoost Huber vs served | +0.145 % | 0.106 |

**Decision:** no candidate meets the rule — the served forecast is unchanged.
