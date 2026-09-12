# Limitations (Chapter 6 material — state these before the examiner does)

## 1. Market unpredictability is the dominant factor
Daily returns of liquid assets are close to a martingale difference sequence: the lag-1 autocorrelation of returns in the training data is −0.08 (BTC), −0.04 (Gold), +0.00 (Silver) (`results/eda_summary.csv`). Under the weak-form efficient-market hypothesis, any predictable component is small, unstable and quickly arbitraged away. The project's central empirical question is therefore *whether* a model beats the random walk at all, and a negative or marginal answer is a legitimate scientific result, not a failed project.

## 2. Small sample
Only ~3 years of daily data were stored (BTC 1,067 usable days; Gold/Silver 724 exchange days), giving ~500–750 training windows and a test set of 111–161 days. Consequences:
* Standard errors are large: with 111 test days, a directional-accuracy of 56 % is not distinguishable from 50 % (binomial p ≈ 0.1). This is why p-values and fold-wise std are reported everywhere.
* Deep sequence models (GRU/LSTM) are data-hungry; their under-performance here partly reflects the sample size.
* `config.DATA_START_DATE = 2018-01-01` and `make collect-data` extend the history 2–3× once Yahoo Finance is not rate-limiting; the whole pipeline is date-driven and re-runs unchanged. Yahoo returned HTTP 429 during this work, so the stored data was used.

## 3. Regime shift in the test period
The test window (Feb–Jul 2026) is *not* like the training window: Gold trades at $3,986–5,294 vs a training range of $1,817–3,644; Silver's 30-day volatility exceeds every value seen in training on 89 % of test days; Bitcoin fell ~15 % in early June 2026. Volatility-based features are out of their training range for Gold/Silver (`volatility_30d`, `atr_norm`). Stationary features reduce but do not remove this problem. A single 5-month test window is also just one draw from a non-stationary process — the walk-forward folds give the more reliable picture.

## 4. Data quality
* Gold/Silver are **front-month futures** (GC=F, SI=F); the price series contains roll effects and the yfinance volume column is unusable (dropped).
* Yahoo Finance is an unofficial, free source; BTC-USD is a composite index that can differ from any exchange's price.
* External series are aligned by forward fill; on holidays the "macro return" is a stale zero.
* Fear & Greed is a proprietary composite whose formula changed over time.

## 5. Modelling choices
* One-day horizon only; multi-day and volatility forecasting were out of scope.
* Point forecasts only. The ±1 RMSE band shown in the dashboard is an ex-post error band, not a calibrated predictive interval.
* Hyper-parameter grids are small by design (seconds–minutes per model); a larger search could change the ranking marginally but not the conclusion about the random walk.
* Model selection uses mean fold RMSE of the return; a practitioner might select on directional accuracy or Sharpe instead — those numbers are reported too.
* The stacked ensemble is a linear blend; more complex meta-learners were not explored.
* Random Forest / boosting see only day-*t* features (plus lags); recurrent models see 30 days. Whether more history helps is answered empirically (it did not, here).

## 6. Evaluation caveats
* Directional accuracy excludes exactly-zero-return days (rare after removing synthetic weekends; 0 in the stored test sets).
* The strategy backtest is a stylised long/flat rule with a flat 10 bps cost — no slippage, no shorting, no position sizing, no financing costs, and no compounding across assets.
* The Diebold–Mariano test assumes covariance-stationary loss differentials; with fat-tailed returns the small-sample behaviour is approximate.

## 7. What the system does NOT do
It does not guarantee or promise future prices, does not account for news, regulation, exchange outages or macro shocks, and is not financial advice. Its value is methodological: a leakage-free, statistically tested comparison plus a transparent demonstration system.
