# Feature Dictionary

All features are computed in `src/data/preprocessing.py::DataCleaner` at trading day *t* from data available at the close of day *t*. `P` = close, `H`/`L` = high/low, `V` = volume, `r_t = ln(P_t / P_{t−1})`.

## Model inputs (stationary)

| # | Feature | Formula | Purpose | Assets |
|---|---|---|---|---|
| 1 | `log_return` | `ln(P_t / P_{t−1})` | today's return; the strongest single predictor of short-term mean reversion / momentum | all |
| 2–5 | `return_1d`, `return_2d`, `return_5d`, `return_10d` | `r_{t−k}` | lagged returns — lets the model see the recent path without a sequence model | all |
| 6–7 | `volatility_10d`, `volatility_30d` | `std(r_{t−w+1..t})` | volatility clustering — the most robust stylised fact of financial returns; large moves follow large moves | all |
| 8 | `RSI` | Wilder RSI(14) on `P` | bounded 0–100 momentum oscillator; overbought/oversold | all |
| 9 | `ADX` | ADX(14) from `H,L,P` | trend-strength (0–100), regime indicator | all |
| 10 | `ROC` | `(P_t / P_{t−12} − 1)·100` | 12-day rate of change | all |
| 11 | `ema14_ratio` | `P_t / EMA_14 − 1` | distance from short trend, scale-free | all |
| 12 | `ema30_ratio` | `P_t / EMA_30 − 1` | distance from medium trend | all |
| 13 | `macd_norm` | `(EMA_12 − EMA_26) / P_t` | MACD line normalised by price (raw MACD grows with the price level) | all |
| 14 | `macd_hist_norm` | `(MACD − Signal_9) / P_t` | MACD histogram normalised | all |
| 15 | `bb_pctb` | `(P_t − BB_lower) / (BB_upper − BB_lower)` | Bollinger %B: position inside the 20-day ±2σ band | all |
| 16 | `bb_width` | `(BB_upper − BB_lower) / BB_mid` | band width: volatility regime | all |
| 17 | `atr_norm` | `ATR_14 / P_t` | average true range relative to price | all |
| 18 | `hl_range` | `(H_t − L_t) / P_t` | today's intraday range (Parkinson-style volatility proxy) | all |
| 19 | `sp500_return` | `ln(SPX_t / SPX_{t−1})` (ffilled to asset calendar) | risk-on/risk-off | all |
| 20 | `vix_return` | `ln(VIX_t / VIX_{t−1})` | change in implied volatility / fear | all |
| 21 | `dxy_return` | `ln(DXY_t / DXY_{t−1})` | US-dollar strength; gold/silver are priced in USD | Gold, Silver |
| 22 | `oil_return` | `ln(WTI_t / WTI_{t−1})` | commodity co-movement / inflation proxy | Gold, Silver |
| 23 | `tnx_return` | `ln(TNX_t / TNX_{t−1})` | 10-year yield change; opportunity cost of holding gold | Gold, Silver |
| 24 | `gold_return` | `ln(GC_t / GC_{t−1})` | gold's same-day return; silver follows gold | Silver |
| 25 | `fear_greed` | alternative.me index 0–100 | crypto sentiment | Bitcoin |
| 26–27 | `dow_sin`, `dow_cos` | `sin/cos(2π·weekday/7)` | 7-day crypto market has documented weekend effects | Bitcoin |
| 28 | `log_volume_change` | `clip(ln(V_t / V_{t−1}), −3, 3)` | attention / activity spikes | Bitcoin |

Counts: Bitcoin 24, Gold 23, Silver 24.

## Kept for charts only (never model inputs — `config.LEVEL_COLUMNS`)
`open, high, low, price, volume, EMA_14, EMA_30, BB_Mid, BB_Upper, BB_Lower, ATR, MACD, MACD_Signal`

Why excluded: they are price levels. Gold's test period (Feb–Jul 2026, $3,986–5,294) lies entirely above its training range ($1,817–3,644), so a level feature would be out-of-distribution on every test day; tree models cannot extrapolate and would return the "highest price ever seen" leaf.

## Features deliberately NOT used
* **Futures volume for Gold/Silver** — yfinance reports front-month contract volume, which collapses and jumps at contract rolls (e.g. 23,155 → 1,025 on consecutive days in the raw file).
* **Day-of-week for metals** — after removing the synthetic weekend rows there is no strong calendar effect to justify it.
* **VWAP, BB_Mid, SMA** — redundant with EMA ratios / %B.
* **Sentiment/news for metals** — no free, reliable daily source.

## Target
`y_t = log_return_{t+1} = ln(P_{t+1} / P_t)` — created only inside `create_sequences` as the next row's `log_return`; the window never contains that row.
