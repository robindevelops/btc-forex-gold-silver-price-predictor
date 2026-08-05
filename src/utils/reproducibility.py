"""
Reproducibility utilities for setting random seeds across all frameworks.
"""

import random
import os
import numpy as np


def set_all_seeds(seed: int = 42) -> None:
    """
    Set seeds for all random number generators to ensure reproducibility.

    Covers: Python stdlib, NumPy, PYTHONHASHSEED, and TensorFlow (if available).

    Args:
        seed: The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Lazy import to avoid triggering TF at module load time
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except (ImportError, OSError, PermissionError):
        pass
