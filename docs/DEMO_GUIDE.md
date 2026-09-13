# FYP Demonstration Guide

A 10-minute walkthrough for evaluators. Start the dashboard with `make serve` (http://localhost:8501).
Every number shown is produced by the pipeline from real data; nothing is hard-coded.

**Before demo day:** `data/` (raw, processed, models) is git-ignored. Run the demo on the machine that ran the pipeline, or copy `data/` and `results/` alongside the repository — otherwise run `make preprocess train stack evaluate` first (≈ 5 min without tuning, using the committed `results/tuning/best_params.json`).

## The story in one sentence
> *Historical data → features → model training on the past → prediction on days the model never saw → actual value → error → evaluation.*

## Step-by-step

| Step | Where | What to say |
|---|---|---|
| 1. Pick an asset | Sidebar → *Target Asset* = Bitcoin | "Daily Yahoo Finance data from 2018; Gold and Silver are exchange days only." |
| 2. Show the data | Tab **Forecast & Indicators** | Point at the shaded regions: green = training (≤ 2025-09-10), yellow = validation (≤ 2026-02-17), red = **unseen test**. Toggle RSI / MACD / Bollinger to show the feature families. |
| 3. Show what the model learned | Tab **Model Performance** → feature-importance figure, loss curves | "The tuner chose small, regularised models; volatility and trend features dominate." |
| 4. **Predict a day** | Tab **Predict a Day (unseen test)** → choose a date → *Generate Prediction* | "The model sees data up to this day only. Predicted close, then the actual close is revealed, the error in $ and %, and whether the direction was right." Point at the diamond/circle on the actual-vs-predicted chart. Try 2–3 dates (a calm day, the June 2026 drop). |
| 5. All models on that day | same tab, table below | "Ten models were compared; the served one was chosen on validation data, not on this test day." |
| 6. Prediction history | Tab **Prediction History** | "Every unseen day, predicted vs actual, error and hit/miss; MAE and hit-rate at the top; downloadable." |
| 7. How good is it, honestly | Tab **Model Performance** | Walk-forward table (selection), test table (evaluated once), the *Honest reading* box, regime table, experiments expander (E1–E4). |
| 8. Live forecast | Tab **Forecast & Indicators** → *Run AI Prediction* | Tomorrow's forecast with the ±1 RMSE band and the disclaimer. |
| 9. Methodology | Tab **Methodology & Models** | Split, features, models, served-model parameters, model inventory. |

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
| How accurate is it? | Prediction-history KPIs (MAE %, hit-rate) and the test table: MAE ≈ 1.3–2.5 % of price — the same as the random walk's, because that is the asset's daily volatility — direction ≈ 47–55 %, not significantly better than the random walk (DM p-values). Never quote MAPE as "accuracy": 100 − MAPE is what a zero forecast scores too. |
| Didn't an earlier version show 62 % for Silver? | Yes — and the final audit traced it to a close-time leak (futures settle 13:30 ET, the macro closes are later). Fixing it removed the effect (`docs/RESULTS.md §2`). Finding and closing that leak *is* a result. |
| Why this model? | Lowest walk-forward RMSE on train+val folds (Model Performance tab, first table); all alternatives are within 0.5 %, so say plainly that the ranking is not decisive. |
| Can it beat the market? | No — and the system says so on every forecast. The contribution is the validated end-to-end pipeline and the honest evaluation. |
| What did more data / other features change? | Experiments expander: E1 (data size), E2 (ablation), E3 (horizon), E4 (volatility). |
