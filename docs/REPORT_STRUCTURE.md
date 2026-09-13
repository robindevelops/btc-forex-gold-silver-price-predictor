# FYP Report Structure — what goes where

Each bullet names the artefact in this repository that supplies the content.

## Chapter 1 — Introduction
* **Problem.** Next-day price forecasting for Bitcoin, Gold and Silver; why it is hard (efficient markets, noise ≫ signal). `docs/METHODOLOGY.md §1`.
* **Motivation.** Decision support for three asset classes with different drivers (crypto sentiment vs macro).
* **Objectives.** (1) build a leakage-free pipeline; (2) compare statistical, tree-based and deep models against the random-walk baseline under walk-forward validation; (3) quantify significance; (4) deliver a working prediction system. State explicitly that objective (2) may have a negative answer.
* **Scope.** Daily data, 1-day horizon, three assets, free data sources; no intraday, no order-book, no news NLP.

## Chapter 2 — Literature Review
* Efficient-market hypothesis and random-walk model (Fama 1970; Malkiel).
* Stylised facts of returns: fat tails, volatility clustering, near-zero autocorrelation (Cont 2001) — link to `results/eda_summary.csv` / `figures/eda_*.png`.
* ARIMA/GARCH vs ML for financial forecasting; gradient boosting on tabular features; LSTM/GRU for sequences; the reproducibility critique (most published "LSTM beats everything" results use leaky evaluation).
* Evaluation methodology literature: walk-forward validation, Diebold–Mariano (1995), directional-accuracy tests (Pesaran–Timmermann).
* Gold drivers (USD, real yields), silver–gold ratio, Bitcoin sentiment indices.

## Chapter 3 — Methodology
* Dataset table and cleaning — `docs/METHODOLOGY.md §2`, `figures/price_history.png`.
* EDA — `results/eda_summary.csv`, `figures/eda_return_distributions.png`, `figures/eda_volatility_clustering.png`, `figures/eda_*_acf_pacf.png`.
* Target definition and reconstruction formula — §1.
* Feature engineering — `docs/FEATURES.md` (table reproduced in the report).
* Split, scaling, sequences — §4 (include the date table).
* Models — §5 with the architecture/hyper-parameter table from `results/tuning/best_params.json`.
* Validation — §6 with the fold diagram.
* Metrics — §7 (define MAE, RMSE, MAPE, R², directional accuracy, DM test).
* Leakage checklist — §9 (examiners like this table).

## Chapter 4 — Implementation
* System architecture diagram: `data/raw → preprocessing → build_dataset → {tune, train, stack} → backtesting → results → {Streamlit, FastAPI}`.
* Module map (`README.md` project structure).
* Reproducibility: `make pipeline`, seeds, pinned requirements, CI, 25 tests (`tests/`).
* Application: Streamlit tabs and the FastAPI endpoints with screenshots.

## Chapter 5 — Results
* Experiment 1 — baselines (naive, mean, ARIMA): `results/FINAL_RESULTS.md`.
* Experiment 2 — original models re-evaluated correctly: `docs/AUDIT_SUMMARY.md` table.
* Experiment 3 — tuned models under walk-forward validation: `results/cv_results.csv`, `figures/model_comparison_cv.png`, tuning curves in `results/tuning/*.csv`.
* Experiment 4 — untouched test set: `results/final_test_results.csv`, `figures/model_comparison_test.png`, per-asset `figures/<asset>_actual_vs_predicted.png`, `_residuals.png`, `_strategy.png`.
* Experiment 5 — analysis: feature importance (`figures/<asset>_feature_importance.png`), over-fitting gap (`figures/overfitting_gap.png`), loss curves, stacked-ensemble weights (`results/stacking/*.json`).
* Experiment 6 — the close-time leak found by the final audit: `results/archive_sameday_macro_leak/` (same-day macro features gave Silver 62 % directional accuracy) vs the corrected results in `results/`; `docs/RESULTS.md §2`. Present it as a finding: a few hours of look-ahead is enough to manufacture a "significant" result on daily data.
* Final table: `docs/RESULTS.md`.

## Chapter 6 — Discussion
* Findings: which model was selected per asset and why; whether it beats the random walk (DM p-values); directional accuracy and its significance; economic value (strategy vs buy-and-hold).
* Strengths: leakage-free methodology, multiple baselines, statistical testing, reproducibility, working system.
* Weaknesses / limitations: `docs/LIMITATIONS.md`.
* Why the original R² ≈ 0.95 was an illusion (one paragraph + the price-vs-return plot).

## Chapter 7 — Conclusion & Future Work
* Longer history (`DATA_START_DATE`), multi-day horizons, volatility (GARCH) targets, probabilistic forecasts, news/NLP sentiment, intraday data, regime-aware models, proper transaction-cost modelling.

## Appendices
* Full feature table, hyper-parameter grids (`src/training/tune_models.py::GRIDS`), per-fold results, viva Q&A (`docs/VIVA_QA.md`).
