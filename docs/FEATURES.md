# Feature Dictionary

All features are computed in `src/data/preprocessing.py::DataCleaner` at trading day *t* from data available **at the moment `P_t` is observed** — the 00:00 UTC bar close for Bitcoin, the 13:30 ET COMEX settlement for Gold/Silver (`docs/METHODOLOGY.md §2`, "close-time rule"). `P` = close, `H`/`L` = high/low, `V` = volume, `r_t = ln(P_t / P_{t−1})`. Rows marked **†** are lagged one session for the metals because Yahoo's futures High/Low extend past the settlement; rows marked **‡** use the previous trading day's value for the metals because those markets close after the 13:30 settlement.

## Model inputs (stationary)

| # | Feature | Formula | Purpose | Assets |
|---|---|---|---|---|
| 1 | `log_return` | `ln(P_t / P_{t−1})` | today's return; the strongest single predictor of short-term mean reversion / momentum | all |
| 2–5 | `return_1d`, `return_2d`, `return_5d`, `return_10d` | `r_{t−k}` | lagged returns — lets the model see the recent path without a sequence model | all |
| 6–7 | `volatility_10d`, `volatility_30d` | `std(r_{t−w+1..t})` | volatility clustering — the most robust stylised fact of financial returns; large moves follow large moves | all |
| 6a | `return_20d` | `Σ r_{t−19..t}` | one-month momentum / reversal | all |
| 6b–6d | `rv_1d`, `rv_5d`, `rv_22d` | `sqrt(mean r² over 1 / 5 / 22 days)` | HAR-style realised volatilities (daily, weekly, monthly) | all |
| 6e | `ewma_vol` | RiskMetrics EWMA σ (λ = 0.94) | exponentially weighted volatility | all |
| 8 | `RSI` | Wilder RSI(14) on `P` | bounded 0–100 momentum oscillator; overbought/oversold | all |
| 9 | `ADX` † | ADX(14) from `H,L,P` (previous session for metals) | trend-strength (0–100), regime indicator | all |
| 10 | `ROC` | `(P_t / P_{t−12} − 1)·100` | 12-day rate of change | all |
| 11 | `ema14_ratio` | `P_t / EMA_14 − 1` | distance from short trend, scale-free | all |
| 12 | `ema30_ratio` | `P_t / EMA_30 − 1` | distance from medium trend | all |
| 13 | `macd_norm` | `(EMA_12 − EMA_26) / P_t` | MACD line normalised by price (raw MACD grows with the price level) | all |
| 14 | `macd_hist_norm` | `(MACD − Signal_9) / P_t` | MACD histogram normalised | all |
| 15 | `bb_pctb` | `(P_t − BB_lower) / (BB_upper − BB_lower)` | Bollinger %B: position inside the 20-day ±2σ band | all |
| 16 | `bb_width` | `(BB_upper − BB_lower) / BB_mid` | band width: volatility regime | all |
| 17 | `atr_norm` † | `ATR_14 / P_t` (previous session for metals) | average true range relative to price | all |
| 18 | `hl_range` † | `(H_t − L_t) / P_t` (BTC); `(H_{t−1} − L_{t−1}) / P_{t−1}` (metals) | intraday range (Parkinson-style volatility proxy) | all |
| 19 | `sp500_return` ‡ | `ln(SPX_t / SPX_{t−1})` (BTC); `ln(SPX_{t−1} / SPX_{t−2})` (metals) | risk-on/risk-off | all |
| 20 | `vix_return` ‡ | `ln(VIX_t / VIX_{t−1})` (BTC); previous day for metals | change in implied volatility / fear | all |
| 21 | `dxy_return` ‡ | `ln(DXY_{t−1} / DXY_{t−2})` | US-dollar strength; gold/silver are priced in USD | Gold, Silver |
| 22 | `oil_return` ‡ | `ln(WTI_{t−1} / WTI_{t−2})` | commodity co-movement / inflation proxy | Gold, Silver |
| 23 | `tnx_return` ‡ | `ln(TNX_{t−1} / TNX_{t−2})` | 10-year yield change; opportunity cost of holding gold | Gold, Silver |
| 24 | `gold_return` | `ln(GC_t / GC_{t−1})` | gold's same-day return (same settlement window as silver); silver follows gold | Silver |
| 25 | `fear_greed` | alternative.me index 0–100 | crypto sentiment | Bitcoin |
| 26–27 | `dow_sin`, `dow_cos` | `sin/cos(2π·weekday/7)` | 7-day crypto market has documented weekend effects | Bitcoin |
| 28 | `log_volume_change` | `clip(ln(V_t / V_{t−1}), −3, 3)` | attention / activity spikes | Bitcoin |

Counts: Bitcoin 29, Gold 28, Silver 29. (Added in the improvement phase: `return_20d`, `rv_1d`, `rv_5d`, `rv_22d`, `ewma_vol`; see `results/experiments/E2_feature_ablation.csv` for their measured effect.)

### Close-time rule (why † and ‡ exist)
Yahoo's daily close for GC=F / SI=F is the 13:25–13:30 ET COMEX **settlement**, not the 17:00 ET end of the Globex session (verified against intraday data, see `docs/METHODOLOGY.md`). Anything that is only known after 13:30 — the day's full High/Low, and the S&P 500 (16:00), VIX (16:15), 10-year yield (~15:00), DXY (17:00) and WTI (14:30) closes — falls *inside* the target interval `ln(P_{t+1}/P_t)` and would be look-ahead if used at row *t*. The metals therefore use the previous session's values for those rows. Bitcoin's bar closes at 00:00 UTC, after every US close, so it keeps same-day values. The first version of this project used same-day values for the metals; that leak alone produced an apparent 62 % directional accuracy for Silver (`docs/RESULTS.md §2`).

## Kept for charts only (never model inputs — `config.LEVEL_COLUMNS`)
`open, high, low, price, volume, EMA_14, EMA_30, BB_Mid, BB_Upper, BB_Lower, ATR, MACD, MACD_Signal`

Why excluded: they are price levels. Gold's test period (Feb–Jul 2026, $3,986–5,294) lies entirely above its training range ($1,817–3,644), so a level feature would be out-of-distribution on every test day; tree models cannot extrapolate and would return the "highest price ever seen" leaf.

## Features deliberately NOT used
* **Futures volume for Gold/Silver** — yfinance reports front-month contract volume, which collapses and jumps at contract rolls (e.g. 23,155 → 1,025 on consecutive days in the raw file).
* **Day-of-week for metals** — after removing the synthetic weekend rows there is no strong calendar effect to justify it.
* **VWAP, BB_Mid, SMA** — redundant with EMA ratios / %B.
* **Sentiment/news for metals** — no free, reliable daily source.

## Target
Served task `return_1d`: `y_t = ln(P_{t+1} / P_t)`, built in `build_dataset()` from the day after the window; the window never contains that row.
Experimental tasks (`config.TASKS`): `return_5d` = Σ r_{t+1..t+5}; `vol_22d` = ln sqrt(mean r²_{t+1..t+22}). Splits are defined by the dates the target covers, so multi-day targets never straddle a split boundary (exact embargo).
