# Limitations (Chapter 6 material — state these before the examiner does)

## 1. Market unpredictability is the dominant factor
Daily returns of liquid assets are close to a martingale difference sequence: the lag-1 autocorrelation of returns in the training data is −0.08 (BTC), −0.04 (Gold), +0.00 (Silver) (`results/eda_summary.csv`). Under the weak-form efficient-market hypothesis, any predictable component is small, unstable and quickly arbitraged away. The project's central empirical question is therefore *whether* a model beats the random walk at all, and a negative or marginal answer is a legitimate scientific result, not a failed project.

## 2. Sample size and the single test window
The improved system uses 8.6 years of daily data (BTC 2,750 training windows, metals 1,874), which removed the worst of the small-sample problem, but the unseen test window is still one 5–7-month period (BTC 207 days, metals 143). Consequences:
* Standard errors remain large for direction: with 143 days, 56 % is not distinguishable from 50 % (binomial p ≈ 0.09); it takes ≈ 58 % to reach p < 0.05.
* **Multiplicity.** Ten models × three assets = 30 test-set rows are reported. Under the null hypothesis, one of them reaching p ≈ 0.02 is expected by chance (Bonferroni-adjusted threshold ≈ 0.0017), so a single "significant" cell in the test table is not evidence of skill unless it is also visible on the walk-forward folds.
* A result that holds in one window may not hold in the next; the walk-forward folds (four blocks over 2019–2026) are the more reliable picture, and there no model is distinguishable from the random walk.
* Deep sequence models (GRU/LSTM) remain the weakest family even with 3× the data.

## 2b. A leak that was found and fixed — and what it teaches
The first evaluation of the improved system reported 62 % directional accuracy for Silver (DM p = 0.002). The final audit traced it to a close-time misalignment: Yahoo's daily close for GC=F/SI=F is the 13:30 ET COMEX *settlement*, while the same-day S&P 500 / VIX / yield / DXY / WTI returns used as features are fixed 1–3.5 hours later — inside the target interval. Lagging those features (and the High/Low-based ones) by one session removed the effect (`docs/RESULTS.md §2`). The lesson is stated in the report: "backward-looking by date" is not the same as "known at the moment the target starts", and a leak of a few hours is enough to manufacture a significant result on daily data.

## 2c. The before/after comparison reads the test set a second time
`src/experiments/before_after.py` scores the archived 3-year system and the final system on identical unseen days. It is a *comparison of two already-frozen systems*, not a selection step — nothing was chosen or tuned on it — but it is the one place where test-set rows are read outside `backtesting.py`, and it is declared as such.

## 3. Regime shift in the test period
The test window (Feb–Sep 2026) is *not* like the training window: Gold trades at $3,986–5,294 vs a training range of $1,817–3,644; Silver's 30-day volatility exceeds every value seen in training on 89 % of test days; Bitcoin fell ~15 % in early June 2026. Volatility-based features are out of their training range for Gold/Silver (`volatility_30d`, `atr_norm`). Stationary features reduce but do not remove this problem. A single 5-month test window is also just one draw from a non-stationary process — the walk-forward folds give the more reliable picture.

## 4. Data quality
* Gold/Silver are **front-month futures** (GC=F, SI=F); the price series contains roll effects and the yfinance volume column is unusable (dropped).
* Yahoo Finance is an unofficial, free source; BTC-USD is a composite index that can differ from any exchange's price.
* External series are aligned by forward fill on the asset's calendar; when the asset trades on a day the external market does not, the previous external return is repeated (a stale value, not a fresh one). For the metals the macro features are one day older than the price by construction (close-time rule).
* Fear & Greed is a proprietary composite whose formula changed over time.

## 5. Modelling choices
* One-day horizon is served; the 5-day-return and 22-day-volatility targets were evaluated as experiments only (E3, E4) and not productised.
* Point forecasts only. The ±1 RMSE band shown in the dashboard is an ex-post error band, not a calibrated predictive interval.
* Hyper-parameter grids are small by design (seconds–minutes per model); a larger search could change the ranking marginally but not the conclusion about the random walk.
* Model selection uses mean fold RMSE of the return; a practitioner might select on directional accuracy or Sharpe instead — those numbers are reported too.
* The stacked ensemble is a linear blend; more complex meta-learners were not explored.
* Random Forest / boosting see only day-*t* features (plus lags); recurrent models see 30 days. Whether more history helps is answered empirically (it did not, here).

## 6. Evaluation caveats
* Directional accuracy excludes exactly-zero-return days (rare after removing synthetic weekends; 0 in the stored test sets).
* The strategy backtest is a stylised long/flat rule with a flat 10 bps cost — no slippage, no shorting, no position sizing, no financing costs, and no compounding across assets.
* The Diebold–Mariano test assumes covariance-stationary loss differentials; with fat-tailed returns the small-sample behaviour is approximate.

## 6b. Data-vendor revisions
Yahoo Finance revised a handful of historical closes between the two downloads (mean < 0.005 %, max 0.75 % on one Bitcoin day). The before/after comparison therefore reports each system against the naive forecast computed on its own data.

## 7. What the system does NOT do
It does not guarantee or promise future prices, does not account for news, regulation, exchange outages or macro shocks, and is not financial advice. Its value is methodological: a leakage-free, statistically tested comparison plus a transparent demonstration system.
