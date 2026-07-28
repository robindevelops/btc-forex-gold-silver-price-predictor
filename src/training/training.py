"""
⚠️ DEPRECATED — DO NOT USE ⚠️

This module was the original training script but was disabled in Week 1
due to critical data leakage (scaler was fit on entire dataset including test data).

Use instead:
  - src/training/train_final_btc.py
  - src/training/train_final_gold.py
  - src/training/train_final_silver.py

These scripts implement the correct 2-phase training with leakage-free evaluation.
"""

raise ImportError(
    "training.py is PERMANENTLY DEPRECATED due to data leakage. "
    "Use train_final_btc.py, train_final_gold.py, or train_final_silver.py instead."
)
