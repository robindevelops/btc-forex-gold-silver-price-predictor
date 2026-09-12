# Viva Preparation — questions and answers grounded in this implementation

Numbers below are from `results/FINAL_RESULTS.md` and `results/cv_results.csv`.

### Why did you choose this model?
The served model per asset is chosen automatically by 4-fold expanding-window walk-forward validation — lowest mean RMSE of the predicted return on the train+val folds (`src/evaluation/backtesting.py`). For all three assets that is **CatBoost** (gradient-boosted trees, learning-rate 0.01, depth 3 for Bitcoin/Silver and 5 for Gold), but Ridge, LightGBM and the historical-mean forecast are within 0.5 % of it, and the random walk is within 1 %. I compared nine models plus a stacked ensemble; the choice is data-driven, not a preference. If asked *why gradient boosting over LSTM*: with 500–750 training windows the recurrent models over-fit (validation R² −0.10 to −0.22, worst on every asset), while shallow regularised trees are the most data-efficient family and give feature importance for interpretation.

### Why did you choose these features?
Every feature has a financial rationale and passes a no-look-ahead test (`docs/FEATURES.md`, `tests/test_features.py`). Three groups: (1) the return path — today's and lagged log returns, rolling volatility (volatility clustering is the most robust stylised fact); (2) standard technical state expressed as *scale-free ratios* — RSI, ADX, ROC, MACD/price, price/EMA−1, Bollinger %B and width, ATR/price, high–low range; (3) exogenous drivers — S&P 500 and VIX returns for all, DXY/oil/10-year-yield returns for the metals, gold's return for silver, Fear & Greed, day-of-week and volume change for Bitcoin. Raw price levels are excluded because they are non-stationary and the test period lies entirely outside the training range for Gold and Silver.

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
Honestly: **it is not, by a statistically significant margin.** On the untouched test set the served CatBoost is within 0.3–1.5 % of the random walk's return-RMSE with Diebold–Mariano p = 0.15–0.57; on walk-forward validation it is 0.3–1.1 % better for Gold/Silver and 0.7 % worse for Bitcoin — all far inside the fold-to-fold standard deviation. Models that deviate more from the drift (ARIMA, GRU, LSTM) are *significantly worse*. That is the scientific result of the project: at a one-day horizon with this data, the best achievable point forecast is approximately the unconditional drift, consistent with weak-form market efficiency. The project's value is the rigorous, leakage-free demonstration of that fact and a system that reports it honestly.

### What does MAE mean?
Mean absolute error — the average size of the miss in the target's units. In USD: $1,097 for Bitcoin (1.6 % of price), $61.7 for Gold, $1.97 for Silver. Robust to outliers.

### What does RMSE mean?
Root mean squared error — like MAE but squares the errors first, so large misses dominate. It is the quantity the models minimise (MSE loss) and the one the Diebold–Mariano test compares. In return space: 0.0207 (BTC), 0.0175 (Gold), 0.0342 (Silver) — essentially the standard deviation of daily returns, which is what a zero forecast scores.

### What does R² mean?
1 − (sum of squared errors / sum of squared deviations from the mean). R² = 0 is "as good as predicting the mean"; negative is worse than the mean. In return space all models are between −0.18 and +0.01. I do **not** report R² on prices because the random walk scores 0.95 there — the original project reported that number and it was misleading.

### What does your "accuracy" actually mean?
"Directional accuracy" = the percentage of test days with a non-zero move on which the sign of the predicted return matched the sign of the actual return, with a one-sided binomial p-value against 50 %. It is not classification accuracy of a classifier and the zero-return baseline has no direction, so it is shown as "—". Values are 42–53 % and none is significant (smallest p = 0.26). A previous version computed it incorrectly (counting flat synthetic days and giving the naive model 0 %).

### Why might Bitcoin be harder to predict than Gold?
Bitcoin's daily volatility is 2.5 % vs 1.0 % for Gold (training window), with fatter tails (excess kurtosis 2.3 vs 1.4), 24/7 trading and sentiment-driven flows; the noise floor is simply higher, so any small predictable component is a smaller fraction of the variance. Gold has identifiable macro drivers (USD, real yields) — but in this sample they were not enough to beat the random walk either. Silver combines gold's drivers with industrial demand and had the largest regime shift in the test period (volatility 3× training), which is why every model's validation RMSE is ~3× its training RMSE there.

### What are the limitations of your model?
`docs/LIMITATIONS.md`: efficient markets (the dominant one), three years of data (Yahoo rate-limited a longer download; the pipeline supports it), a single 5-month test window with a strong regime shift, unofficial data (front-month futures, composite BTC index), a one-day point forecast with an ex-post error band rather than a calibrated interval, small tuning grids, and a stylised strategy backtest.

### Can your model guarantee future prices?
No. The dashboard shows a ±1 RMSE band (e.g. ±$1,420 around a $63,400 Bitcoin forecast) that is an order of magnitude wider than the predicted move, and states that no model beat the random walk significantly. It is a decision-support research prototype, not advice.

### Why did the stacked ensemble collapse?
With non-negative Ridge on out-of-fold predictions, the meta-learner assigns weight only to base models that carry out-of-sample signal. On Bitcoin all four weights are exactly 0 (intercept only). That is the stack correctly reporting that there is nothing to combine — the original project served exactly such a constant model as its "primary model" without noticing.

### What did the original project get wrong, and what did you change?
`docs/AUDIT_SUMMARY.md` and `docs/FIX_PLAN.md`. Headlines: USD metrics reconstructed from the wrong column; 30 % fabricated weekend rows for metals; hyper-parameters selected on the test set; early stopping on training data; price-level features out of range; a constant-predicting ensemble served as the main model; a broken inference path; misleading R² on prices; stale contradictory results. All fixed, tested and re-run.

### If you had more time?
Longer history (2018+), volatility targets where the features clearly carry information, multi-day horizons, quantile/probabilistic forecasts scored by calibration, intraday data, news NLP for the metals, and regime-switching models.
