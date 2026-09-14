# FYP Demonstration Guide

A 10-minute walkthrough for evaluators. Start the dashboard with `make serve` (http://localhost:8501).
Every number shown is produced by the pipeline from real data; nothing is hard-coded.

**Before demo day:** `data/` (raw, processed, models) is git-ignored. Run the demo on the machine that ran the pipeline, or copy `data/` and `results/` alongside the repository — otherwise run `make preprocess train stack evaluate` first (≈ 15 min without tuning, using the committed `results/tuning/best_params.json`). Press *Run prediction* in the afternoon rather than early morning: Yahoo publishes Bitcoin's previous-day bar a few hours after 00:00 UTC, and until then the Bitcoin forecast starts from the day before (the dashboard says so).

## The story in one sentence
> *Historical data → features → six models trained on the past → one combined prediction on days the models never saw → actual value → error → evaluation.*

## Step-by-step

| Step | Where | What to say |
|---|---|---|
| 1. Pick an asset | Sidebar → *Asset* = Bitcoin (the overview strip under the title states what is predicted, that the served forecast combines six models, and its test reliability) | "Daily Yahoo Finance data from 2018, complete bars only; Gold and Silver are exchange days only. There is no model to choose — all six run and are combined." |
| 2. Show the data | Tab **Forecast** | Point at the shaded regions: green = training (≤ 2025-09-10), yellow = validation (≤ 2026-02-17), red = **unseen test**. Use *Chart overlays* (sidebar) to show the RSI / MACD / Bollinger families the features are built from. |
| 3. Show what the models learned | Tab **Model Performance** → "Which inputs does a member model rely on?" chart and the *Training diagnostics* expander | "The tuner chose small, regularised models; volatility and trend features dominate." |
| 4. **Predict a day** | Tab **Predict a Day (unseen test)** → choose a date → *Generate prediction* | "All six models see data up to this day only. Their combined predicted close, then the actual close is revealed, the error in $ and %, and HIT or FAIL." Point at the diamond/circle on the actual-vs-predicted chart. Try 2–3 dates (a calm day, the June 2026 drop). |
| 5. Each model on that day | same tab, table below | "These six predictions were averaged into the combined one; they agree to a fraction of a percent because daily returns are mostly noise." |
| 6. Test-set history | Tab **Test-Set History** | "Every unseen test day, predicted vs actual, error and HIT/FAIL for the combined forecast; MAE and hit-rate at the top; downloadable." |
| 7. How good is it, honestly | Tab **Model Performance** | Test table (evaluated once, Combined row marked), the *Honest reading* box, the model-vs-random-walk chart, walk-forward table, predicted-vs-actual returns; regimes and experiments in expanders. |
| 8. Live forecast | Sidebar → *Run prediction* (tab **Forecast**) | "All six models run on the last complete bar; the combined next-day forecast, UP/DOWN, the ±1 RMSE band and each model's own prediction." |
| 9. Methodology | Tab **Methodology** | Split, features, models, served-model parameters, model inventory. |

## Figures to put on slides (all in `results/figures/`)
1. `price_history.png` — the three assets with train/val/test shading (the split).
2. `<asset>_actual_vs_predicted.png` — **the main result**: returns (top), scatter (middle), price (bottom) on unseen days.
3. `model_comparison_test.png` — all models vs the random-walk line.
4. `overfitting_gap.png` — train vs validation, with the random walk's own gap.
5. `<asset>_feature_importance.png` — what the model uses.
6. `<asset>_loss_curves.png` — GRU/LSTM early stopping.
7. `eda_return_distributions.png`, `eda_volatility_clustering.png` — why returns, not prices.
8. Dashboard screenshots: `docs/figures/dash_*.png`.

## Questions evaluators ask — and where the answer is on screen
| Question | Answer (and evidence) |
|---|---|
| Did you train on the test period? | No — frozen chronological split; the red region is never used for training, validation or selection (`config.py`, tests). |
| How do we know it isn't memorising? | *Predict a Day* on unseen dates; `overfitting_gap.png`; the served models are small (CatBoost depth 7 / 50 rounds for Bitcoin, depth 3 / 606 rounds at learning-rate 0.01 for Gold, a 32-unit GRU stopped after 9 epochs for Silver) and their validation gap equals the random walk's own gap. |
| What does the model use? | Methodology tab feature list; `docs/FEATURES.md`; feature-importance figure. |
| How accurate is it? | Test-set history KPIs (MAE %, hit-rate) and the test table: MAE ≈ 1.3–2.5 % of price — the same as the random walk's, because that is the asset's daily volatility — direction ≈ 47–54 %, not significantly better than the random walk (DM p-values). Never quote MAPE as "accuracy": 100 − MAPE is what a zero forecast scores too. |
| Didn't an earlier version show 62 % for Silver? | Yes — and the final audit traced it to a close-time leak (futures settle 13:30 ET, the macro closes are later). Fixing it removed the effect (`docs/RESULTS.md §2`). Finding and closing that leak *is* a result. |
| Why this model? | None alone: the served forecast is the equal-weight combination of all six, because on the walk-forward folds they are within 0.5 % of each other (Model Performance tab, second table) — say plainly that the ranking is not decisive and that equal weights cannot be tuned on the test set. |
| Can it beat the market? | No — and the system says so on every forecast. The contribution is the validated end-to-end pipeline and the honest evaluation. |
| What did more data / other features change? | Experiments expander: E1 (data size), E2 (ablation), E3 (horizon), E4 (volatility). |
