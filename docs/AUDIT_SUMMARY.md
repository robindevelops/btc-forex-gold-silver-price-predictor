# Original Audit — Summary (before the fixes)

Full audit was delivered in conversation; this is the condensed record for the report's
"what changed and why" discussion. **Original FYP score: 50 / 100 (Basic FYP, methodology not defensible).**

## What was already good (kept)
* Log-return target with price reconstruction (most student projects predict price and get an illusory R² ≈ 0.99).
* Chronological 70/15/15 split, scaler fitted on train only.
* Backward-looking indicator code (`DataCleaner`), macro + sentiment data integration.
* Two-phase GRU training (epochs from validation → refit on train+val).
* Baselines existed (naive, ARIMA, linear regression, random forest).
* Streamlit dashboard, modular `src/`, config, Docker, FastAPI, tests, logging.

## What was wrong (all fixed — see `docs/FIX_PLAN.md`)
| Problem | Effect | Fix |
|---|---|---|
| `reconstruct_price` used column 0 = `open` instead of `price` | every USD metric and every dashboard forecast off by one intraday move (BTC ≈ $1,416 RMSE from the bug alone) | explicit previous close; regression test |
| Gold/Silver forward-filled to calendar days | 30.7 % fabricated zero-return rows; 47/159 test days fake; naive "directional accuracy 29.6 %" was literally the share of fake days | trading-day calendar |
| Naive forecast beat every model; R² 0.94–0.97 on price shown as skill | misleading conclusions | return-space metrics, DM test, honest reporting |
| Hyper-parameters chosen on the test set (April experiments); never re-tuned after target/features changed | test contamination | walk-forward tuner, test never loaded |
| CatBoost early-stopped on data inside its training set | not regularised | two-phase fit |
| Price-level features (open/high/low/price/EMA/BB/VWAP) MinMax-scaled | test values 1.7–2.4× outside the training range | stationary ratio features |
| Served "stacked ensemble" had coefficients ≈ 0 (BTC exactly [0,0,0]) — a constant predictor; never evaluated on test; `prediction.py` and API crashed (LightGBM given 900 features; `float(dict)`) | the demo could not actually run the configured model | inference rewritten around the CV-selected model; stack kept as an experiment with non-negative weights |
| Directional accuracy metric: `sign(prev−prev)=0` for naive; flat days counted | 0 % / 30 % nonsense values | defined on non-flat days with binomial p-value |
| Five stale result files from the price-target era mixed with new ones | contradictory numbers | archived to `results/archive_pre_fix/` |
| README claimed PatchTST, TFT, Williams %R, CCI, Stochastic, "30+ indicators" | none existed | accurate docs |
| Makefile / retrain / CI referenced non-existent scripts; 4/31 tests failed | not reproducible | rewritten; 25/25 tests pass |

## Final audit (after the improvement phase) — one more leak
| Problem | Effect | Fix |
|---|---|---|
| Close-time misalignment for the metals: Yahoo's GC=F/SI=F daily close is the 13:30 ET **settlement** (verified against intraday bars), but same-day S&P 500 / VIX / 10-y yield / DXY / WTI returns (fixed 14:30–17:00 ET) and the session High/Low were used as day-*t* features | 1–3.5 hours of the target interval leaked into the features; Silver's served LightGBM showed 62 % directional accuracy (DM p = 0.002) that vanished when the macro features were lagged (→ ≈ 53 %); Bitcoin unaffected (its bar closes after the US close) | previous-day macro returns for commodities, High/Low-based features lagged one session, both unit-tested; full pipeline re-run; docs and report corrected (`docs/RESULTS.md §2`) |
| Live sync overwrote frozen raw macro files; `make preprocess` failed outside `retrain.py`; demo look-ahead test did not perturb anything; Keras phase A/B used different procedures; BTC Sharpe annualised with 252 | reproducibility / test strength / minor metric | `data/raw_live/`; `sys.path` bootstrap; real perturbation test; identical phases; 365 periods for crypto |

## Independent re-evaluation of the OLD models (test set, bug corrected), for the record
| Asset | Model | RMSE (ret) | R² (ret) | Dir. Acc | RMSE ($) |
|---|---|---:|---:|---:|---:|
| BTC | Zero-return | 0.02064 | −0.000 | — | 1,421 |
| BTC | old GRU | 0.02390 | −0.341 | 52.2 % | 1,638 |
| BTC | old LightGBM | 0.02247 | −0.186 | 54.7 % | 1,535 |
| BTC | old Stacked | 0.02071 | −0.007 | 47.2 % | 1,427 (pred. std 0.0008 — constant) |
| Gold | Zero-return | 0.01459 | −0.009 | — | 66.79 |
| Gold | old GRU | 0.01595 | −0.206 | 45.5 % | 72.86 |
| Silver | Zero-return | 0.02885 | −0.004 | — | 2.18 |
| Silver | old LightGBM | 0.05434 | −2.564 | 43.8 % | 4.09 |
