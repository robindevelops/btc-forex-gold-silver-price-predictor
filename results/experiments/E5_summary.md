# E5 — final-audit improvement candidates (walk-forward folds only; the test set is never read)

Decision rule: keep a candidate only if it lowers the pooled out-of-fold RMSE for every asset with DM p < 0.05 against the served forecast.

| Asset | Candidate | RMSE vs naive | RMSE vs served | DM p vs served | Dir. Acc % (p) | UP calls % |
|---|---|---:|---:|---:|---:|---:|
| Bitcoin | served: Combined (6 models, all features) | -0.018 % | +0.000 % | — | 50.8 (0.234) | 67 |
| Bitcoin | E5a: Combined (4 tabular models) | -0.109 % | -0.090 % | 0.038 | 50.4 (0.362) | 61 |
| Bitcoin | E5b: Combined (4 tabular models, compact features) | -0.020 % | -0.001 % | 0.988 | 48.8 (0.873) | 58 |
| Bitcoin | E5c: Combined zero-drift | -0.151 % | -0.132 % | 0.027 | 50.5 (0.317) | 48 |
| Gold | served: Combined (6 models, all features) | -0.213 % | +0.000 % | — | 52.5 (0.023) | 74 |
| Gold | E5a: Combined (4 tabular models) | -0.257 % | -0.044 % | 0.499 | 55.1 (0.000) | 86 |
| Gold | E5b: Combined (4 tabular models, compact features) | -0.191 % | +0.022 % | 0.755 | 55.1 (0.000) | 85 |
| Gold | E5c: Combined zero-drift | -0.039 % | +0.174 % | 0.027 | 48.0 (0.949) | 28 |
| Silver | served: Combined (6 models, all features) | -0.058 % | +0.000 % | — | 52.7 (0.016) | 78 |
| Silver | E5a: Combined (4 tabular models) | -0.025 % | +0.033 % | 0.486 | 51.4 (0.140) | 72 |
| Silver | E5b: Combined (4 tabular models, compact features) | +0.093 % | +0.151 % | 0.179 | 52.0 (0.063) | 66 |
| Silver | E5c: Combined zero-drift | -0.026 % | +0.032 % | 0.428 | 49.9 (0.550) | 38 |

E5b detail (each tabular model, compact vs all features; positive = compact is worse):

| Asset | Model | RMSE change | DM p |
|---|---|---:|---:|
| Bitcoin | Ridge | -0.019 % | 0.519 |
| Bitcoin | RandomForest | +0.086 % | 0.359 |
| Bitcoin | LightGBM | +0.272 % | 0.125 |
| Bitcoin | CatBoost | +0.170 % | 0.166 |
| Gold | Ridge | +0.022 % | 0.535 |
| Gold | RandomForest | +0.038 % | 0.491 |
| Gold | LightGBM | +0.222 % | 0.002 |
| Gold | CatBoost | +0.014 % | 0.892 |
| Silver | Ridge | -0.011 % | 0.759 |
| Silver | RandomForest | +0.047 % | 0.584 |
| Silver | LightGBM | +0.137 % | 0.035 |
| Silver | CatBoost | +0.619 % | 0.067 |
