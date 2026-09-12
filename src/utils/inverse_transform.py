"""
Price reconstruction from a predicted log return.

    P̂_{t+h} = P_t · exp(r̂)

`P_t` is the actual close on day t, passed in explicitly (from the unscaled feature frame).
Earlier versions guessed the price column inside the scaled window and picked `open` by
mistake, which corrupted every USD metric; that API is gone.
"""
import numpy as np


def reconstruct_price(log_return, prev_close):
    """
    Args:
        log_return : (n,) real (un-standardised) log-return predictions or true targets
        prev_close : (n,) actual close price on day t (USD)
    Returns:
        (n,) reconstructed USD close for day t+h
    """
    return np.ravel(prev_close) * np.exp(np.ravel(log_return))
