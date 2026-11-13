# Fresh hold-out — frozen models on days newer than the frozen dataset

Run 2026-11-13 18:41. Nothing was refitted; no version of the project had seen these days. Decision rule (fixed in advance): skill only if Combined RMSE(return) < random walk AND Diebold–Mariano p < 0.05.

| Asset | Days (targets) | Forecast | RMSE vs random walk | DM p | Direction % (p) | UP calls % / up days % | MAPE % |
|---|---|---|---:|---:|---:|---:|---:|
| Bitcoin | 61 (2026-09-13 → 2026-11-12) | Combined | -0.51 % | 0.41 | 47.5 (0.70) | 48 / 48 | 1.35 |
| Bitcoin | 61 (2026-09-13 → 2026-11-12) | Naive-Mean | +0.26 % | 0.53 | 47.5 (0.70) | 100 / 48 | 1.35 |
| Bitcoin | 61 (2026-09-13 → 2026-11-12) | Naive | +0.00 % | — | — | 0 / 48 | 1.34 |
| Gold | 44 (2026-09-14 → 2026-11-12) | Combined | +1.34 % | 0.29 | 43.2 (0.85) | 100 / 43 | 1.06 |
| Gold | 44 (2026-09-14 → 2026-11-12) | Naive-Mean | +1.09 % | 0.12 | 43.2 (0.85) | 100 / 43 | 1.06 |
| Gold | 44 (2026-09-14 → 2026-11-12) | Naive | +0.00 % | — | — | 0 / 43 | 1.05 |
| Silver | 44 (2026-09-14 → 2026-11-12) | Combined | +0.99 % | 0.17 | 43.2 (0.85) | 100 / 43 | 1.91 |
| Silver | 44 (2026-09-14 → 2026-11-12) | Naive-Mean | +0.78 % | 0.09 | 43.2 (0.85) | 100 / 43 | 1.90 |
| Silver | 44 (2026-09-14 → 2026-11-12) | Naive | +0.00 % | — | — | 0 / 43 | 1.89 |

**Verdict per asset (rule above):**

- Bitcoin: no skill shown (Combined -0.51 % vs random walk, DM p = 0.41, 61 days; max revision of the live download vs the frozen closes on the shared history: 0.00 %).
- Gold: no skill shown (Combined +1.34 % vs random walk, DM p = 0.29, 44 days; max revision of the live download vs the frozen closes on the shared history: 1.43 %).
- Silver: no skill shown (Combined +0.99 % vs random walk, DM p = 0.17, 44 days; max revision of the live download vs the frozen closes on the shared history: 0.00 %).

Caveats: ≈ 40–60 days give the tests little power (a real but small edge would usually not reach p < 0.05). The live series is a fresh Yahoo download: history can be revised, and GC=F / SI=F follow the front-month contract, so a contract roll can add one artificial daily return. Every single model is in fresh_holdout.csv.
