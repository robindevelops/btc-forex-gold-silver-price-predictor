"""
Price reconstruction from a predicted log return.

    P̂_{t+1} = P_t · exp(r̂_{t+1})

`P_t` is the actual close on day t, passed in explicitly (from the unscaled feature
frame).  Earlier versions guessed the price column inside the scaled window and
picked `open` by mistake, which corrupted every USD metric; that API is gone.
"""
import numpy as np


def reconstruct_price(y_pred_scaled, prev_close, scaler, target_col_idx):
    """
    Args:
        y_pred_scaled : (n,) or (n,1) scaled log-return predictions (or true scaled targets)
        prev_close    : (n,) actual close price on day t  (unscaled, USD)
        scaler        : the fitted MinMaxScaler (used only to unscale the target column)
        target_col_idx: index of the target column inside the scaler
    Returns:
        (n,) reconstructed USD close for day t+1
    """
    lo, hi = scaler.data_min_[target_col_idx], scaler.data_max_[target_col_idx]
    log_ret = np.ravel(y_pred_scaled) * (hi - lo) + lo
    return np.ravel(prev_close) * np.exp(log_ret)
