# Archived results (pre-fix)

These files were produced by earlier versions of the pipeline and are kept **only for the
project history / "what was wrong" discussion**. They must NOT be cited as results.

Why they are invalid:
- `experiment_log.csv`, `final_9model_comparison.csv`, `ensemble_comparison.csv`, `walk_forward_results.csv`,
  `baseline_metrics.csv`: April 2026 runs that predicted **price levels** with 16 features and used the
  **test set for early stopping / hyper-parameter selection**. The R² ≈ 0.94 values are the random-walk illusion.
- `final_performance_table.csv`, `directional_accuracy.png`, `*_stacked_ensemble.png`: August 2026 runs on the
  log-return pipeline, but every model's USD price was reconstructed from the previous day's **open** instead of
  **close** (`reconstruct_price` bug), and Gold/Silver contained ~30% forward-filled weekend rows.

Current, valid results live in `results/` and are produced by `src/evaluation/backtesting.py`.
