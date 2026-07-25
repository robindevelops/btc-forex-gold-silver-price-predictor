"""
Reproducibility utilities for the crypto-forex prediction system.

Centralizes all random seed setting to ensure reproducible experiments.
Import and call set_all_seeds() at the top of every training/experiment script.

Created: Week 1, Day 2 — Audit fix for missing random seeds.
"""

import os
import random
import numpy as np


def set_all_seeds(seed: int = 42):
    """
    Set all random seeds for reproducibility across Python, NumPy, and TensorFlow.

    Must be called BEFORE any model construction or data shuffling.

    Args:
        seed: Integer seed value. Default 42.
    """
    # Python built-in
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Hash seed (affects set/dict ordering in some Python versions)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # TensorFlow (import lazily to avoid forcing TF load in non-training scripts)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    print(f"  🔒 All random seeds set to {seed}")
