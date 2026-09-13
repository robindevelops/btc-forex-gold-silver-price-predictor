# Archived results — same-day macro alignment (LEAK, superseded)

These are the results of the improvement phase **before** the final audit (git commit `aab7e8c`, 2026-09-12).
They are kept only for the "what was wrong" discussion and **must not be cited as results**.

## Why they are invalid
Yahoo's daily close for GC=F / SI=F is the 13:25–13:30 ET COMEX **settlement**, not the 17:00 ET end of the
Globex session (verified against 5-minute bars of the December-2026 contracts: mean |daily close − 13:30 price| =
0.04 % gold / 0.09 % silver, versus 0.3–1 % against the 17:00 price). This pipeline version fed the metals the
**same-day** S&P 500 (16:00 ET), VIX (16:15), 10-year yield (~15:00), DXY (17:00) and WTI (14:30) returns, and the
session High/Low (`hl_range`, `atr_norm`, `ADX`), as day-*t* features — information from **inside** the target
interval `ln(P_{t+1}/P_t)`.

## What it did to the numbers
| Silver, LightGBM (same tuned params) | Test DA | DM p vs naive | R² (ret) |
|---|---:|---:|---:|
| this archive (same-day macro) | 62.2 % (p = 0.002) | 0.002 | +0.044 |
| same model, five macro features removed | 51.7 % | 0.21 | +0.015 |
| same model, macro lagged one day (the fix) | ≈ 53 % | ≈ 0.1 | ≈ +0.015 |

Bitcoin was never affected (its bar closes at 00:00 UTC, after every US close); Gold's numbers were at the
random-walk floor with and without the leak.

The corrected pipeline (`config.EXTERNAL_SAME_DAY`, `config.POST_SETTLEMENT_FEATURES`,
`DataCleaner._align_external`, `DataCleaner.lag_post_settlement_features`) and its results live in `results/`.
See `docs/METHODOLOGY.md` ("close-time rule") and `docs/RESULTS.md §2`.
