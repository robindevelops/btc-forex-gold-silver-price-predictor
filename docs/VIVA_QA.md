# Viva Preparation — questions and answers grounded in this implementation

Numbers below are from `results/FINAL_RESULTS.md` and `results/cv_results.csv`.

### Why did you choose this model?
The served model per asset is chosen automatically by 4-fold expanding-window walk-forward validation — lowest mean RMSE of the predicted return on the train+val folds (`src/evaluation/backtesting.py`). With the full 2018→ data that is **CatBoost for Bitcoin** and **LightGBM for Gold and Silver** (gradient-boosted trees, 15 leaves / depth 7, learning-rate 0.03, early-stopped after 10–50 rounds). All nine models are within 0.3 % of each other and of the random walk on validation, so the choice is data-driven but not decisive; I say so. If asked *why gradient boosting over LSTM*: the recurrent models are the weakest family on every asset even with 2,750 windows, while shallow boosted trees are data-efficient, fast (seconds) and give feature importance.

### Why did you choose these features?
Every feature has a financial rationale and passes a no-look-ahead test (`docs/FEATURES.md`, `tests/test_features.py`). Four groups: (1) the return path — today's, lagged and 20-day cumulative log returns; (2) volatility state — rolling std, HAR realised volatilities (1/5/22-day), EWMA volatility, Bollinger width, ATR/price, high–low range; (3) momentum/trend as *scale-free ratios* — RSI, ADX, ROC, MACD/price, price/EMA−1, Bollinger %B; (4) exogenous drivers — S&P 500 and VIX returns for all, DXY/oil/10-year-yield returns for the metals, gold's return for silver, Fear & Greed, day-of-week and volume change for Bitcoin. A feature-group ablation (E2) shows no group changes walk-forward RMSE by more than ±0.5 %, so the full set is kept for interpretability. Raw price levels are excluded because they are non-stationary.

### Why are you predicting return instead of price?
Price is non-stationary (ADF p = 0.74–0.98); return is stationary (p < 1e-14) — see `results/eda_summary.csv`. A price model gets R² ≈ 0.95 by copying yesterday's price, which is not skill. Predicting `ln(P_{t+1}/P_t)` makes the random-walk baseline explicit (ŷ = 0) and lets R², the Diebold–Mariano test and directional accuracy mean something. The price is recovered exactly as `P_t · exp(ŷ)`.

### How did you prevent data leakage?
Ten specific checks, listed in `docs/METHODOLOGY.md §9`: chronological split by date (no shuffling); scaler fitted on training rows only; the target is the next row's return and is never inside the input window (unit-tested); every rolling/EWM/shift feature is backward-looking (a unit test perturbs the last day and asserts earlier features are unchanged); external series aligned by forward fill only; early stopping never monitors data it trains on; hyper-parameters tuned on walk-forward folds inside train+val; the test set is read by exactly one script; no synthetic rows; price reconstruction verified to be exact. I also found and fixed real leakage-adjacent bugs in the earlier version (test set used for hyper-parameter selection; CatBoost early-stopping on its own training data; reconstruction from the wrong column).

### Why can't you randomly split financial time-series data?
Because a random split puts day *t+1* in the training set while day *t* is in the test set. The model then "predicts" a day whose neighbours it has already seen — regime, volatility level and trend all leak. It also breaks the deployment scenario: in production you only ever have the past. Walk-forward validation reproduces that scenario and yields several independent out-of-sample windows.

### What is overfitting?
A model fitting the noise of the training sample instead of the signal, so training error keeps falling while out-of-sample error rises. Evidence in this project: GRU/LSTM validation loss reaches its minimum at epoch 2–13 and then rises (`figures/<asset>_loss_curves.png`); the tuner chose the most regularised configurations everywhere; and the train-vs-validation gap of every model equals the random walk's own gap, which shows the remaining gap is regime shift, not memorisation (`figures/overfitting_gap.png`).

### What is your baseline?
The zero-return (random-walk) forecast: tomorrow's price = today's. Also the historical-mean (drift) forecast and ARIMA. Any model must beat the random walk to claim skill.

### Why is your model better than the baseline?
**For Bitcoin and Gold: not by a statistically significant margin.** The served models are within ±0.5 % of the random walk's return-RMSE (DM p = 0.42–0.74) and directional accuracy is 47–51 %. **For Silver: yes, on the test window.** The served LightGBM has 62.2 % directional accuracy on 143 unseen days (binomial p = 0.002), R² = +0.044 in return space and a Diebold–Mariano p = 0.002 against the random walk, and the edge holds in every volatility and trend regime of the window. I report it with its caveat: the same model scored 52.8 % on the walk-forward folds, so the test window is unusually favourable and the effect may not persist. Overall the result is consistent with weak-form market efficiency; the project's value is the rigorous, leakage-free demonstration and a system that reports its own error honestly.

### What does MAE mean?
Mean absolute error — the average size of the miss in the target's units. In USD on the unseen test days: $1,071 for Bitcoin (1.5 % of price), $58.0 for Gold (1.3 %), $1.75 for Silver (2.4 %). Robust to outliers.

### What does RMSE mean?
Root mean squared error — like MAE but squares the errors first, so large misses dominate. It is the quantity the models minimise (MSE loss) and the one the Diebold–Mariano test compares. In return space: 0.0208 (BTC), 0.0166 (Gold), 0.0312 (Silver) — essentially the standard deviation of daily returns, which is what a zero forecast scores.

### What does R² mean?
1 − (sum of squared errors / sum of squared deviations from the mean). R² = 0 is "as good as predicting the mean"; negative is worse than the mean. In return space the served models score −0.004 (BTC), +0.008 (Gold) and +0.044 (Silver). I do **not** report R² on prices because the random walk scores 0.95 there — the original project reported that number and it was misleading.

### What does your "accuracy" actually mean?
"Directional accuracy" = the percentage of test days with a non-zero move on which the sign of the predicted return matched the sign of the actual return, with a one-sided binomial p-value against 50 %. It is not classification accuracy of a classifier and the zero-return baseline has no direction, so it is shown as "—". Served models: 46.9 % (BTC, p = 0.83), 51.0 % (Gold, p = 0.43), 62.2 % (Silver, p = 0.002). A previous version computed it incorrectly (counting flat synthetic days and giving the naive model 0 %).

### Why might Bitcoin be harder to predict than Gold?
Bitcoin's daily volatility is 2.5 % vs 1.0 % for Gold (training window), with fatter tails (excess kurtosis 2.3 vs 1.4), 24/7 trading and sentiment-driven flows; the noise floor is simply higher, so any small predictable component is a smaller fraction of the variance. Gold has identifiable macro drivers (USD, real yields) — but in this sample they were not enough to beat the random walk either. Silver combines gold's drivers with industrial demand and had the largest regime shift in the test period (volatility 3× training), which is why every model's validation RMSE is ~3× its training RMSE there.

### What are the limitations of your model?
`docs/LIMITATIONS.md`: efficient markets (the dominant one), a single 5–7-month test window with a strong regime shift, unofficial data (front-month futures, composite BTC index, vendor revisions), a one-day point forecast with an ex-post error band rather than a calibrated interval, small tuning grids, and a stylised strategy backtest.

### Can your model guarantee future prices?
No. The dashboard shows a ±1 RMSE band (e.g. ±$1,420 around a $63,400 Bitcoin forecast) that is an order of magnitude wider than the predicted move, and states that no model beat the random walk significantly. It is a decision-support research prototype, not advice.

### Why did the stacked ensemble collapse?
With non-negative Ridge on out-of-fold predictions, the meta-learner assigns weight only to base models that carry out-of-sample signal. With 3 years of data all four weights were exactly 0 for Bitcoin (intercept only) — the stack correctly reporting there was nothing to combine. With the full 2018→ data the weights are non-zero (Bitcoin: LightGBM 0.39, CatBoost 0.31; Silver: LightGBM 1.1, CatBoost 0.47, GRU 0.56), and the stack matches the best single model on the test set. It is reported as an experiment; the served model is the single validation winner.

### What did the original project get wrong, and what did you change?
`docs/AUDIT_SUMMARY.md` and `docs/FIX_PLAN.md`. Headlines: USD metrics reconstructed from the wrong column; 30 % fabricated weekend rows for metals; hyper-parameters selected on the test set; early stopping on training data; price-level features out of range; a constant-predicting ensemble served as the main model; a broken inference path; misleading R² on prices; stale contradictory results. All fixed, tested and re-run.

### If you had more time?
Longer history (2018+), volatility targets where the features clearly carry information, multi-day horizons, quantile/probabilistic forecasts scored by calibration, intraday data, news NLP for the metals, and regime-switching models.

### What did you improve, and did it help?
Three things, each measured (`docs/RESULTS.md §2–3`, `results/experiments/`): (1) 2.9× more training data (2018→) — Bitcoin validation RMSE −1 to −1.7 %, metals neutral; (2) HAR realised-volatility, EWMA-volatility and 20-day-momentum features — no group moves RMSE by more than ±0.5 %; (3) re-tuning and a split defined by target dates (exact embargo). On identical unseen days the served models improved by 1.9 % (Gold) and 2.8 % (Silver) in return-RMSE and were unchanged for Bitcoin. Experiments that did *not* help and were not kept: a 5-day horizon (E3) and a volatility target (E4, only ~4 % better than an EWMA).

### How does the "Predict a Day" demo prove the model isn't cheating?
`predict_for_date(asset, day)` slices the feature frame to rows ≤ day before scaling and predicting, so nothing after the chosen day can influence the forecast (unit-tested). The actual close is read afterwards only to compute the error. Every day after 2026-02-17 is outside training, validation and model selection.
