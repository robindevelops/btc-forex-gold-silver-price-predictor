import numpy as np

def reconstruct_price(y_pred_scaled, X_seq_scaled, scaler, target_col_idx, price_col_idx=0):
    """
    Inverse transforms the target.
    If target is price, it just inverse transforms it directly.
    If target is log_return, it inverse transforms the log_return,
    inverse transforms the t-1 price from X_seq_scaled,
    and computes: Price_t = Price_{t-1} * exp(log_return_t)
    
    Args:
        y_pred_scaled: array of shape (samples, 1) or (samples,)
        X_seq_scaled: array of shape (samples, seq_len, n_features)
        scaler: fitted MinMaxScaler
        target_col_idx: column index of the target (e.g. 2 for log_return, 0 for price)
        price_col_idx: column index of raw price (typically 0)
        
    Returns:
        reconstructed_price: array of unscaled USD prices at time t
    """
    n_features = scaler.n_features_in_
    
    # Inverse transform the target (could be scaled price or scaled log_return)
    dummy_y = np.zeros((len(y_pred_scaled), n_features))
    dummy_y[:, target_col_idx] = np.array(y_pred_scaled).ravel()
    unscaled_target = scaler.inverse_transform(dummy_y)[:, target_col_idx]
    
    # If the model directly predicted price, we're done
    if target_col_idx == price_col_idx:
        return unscaled_target
        
    # If the model predicted log_return, we must reconstruct the price
    # Get t-1 scaled prices from the last step of the input sequence
    scaled_price_t_minus_1 = X_seq_scaled[:, -1, price_col_idx]
    dummy_x = np.zeros((len(scaled_price_t_minus_1), n_features))
    dummy_x[:, price_col_idx] = scaled_price_t_minus_1
    price_t_minus_1 = scaler.inverse_transform(dummy_x)[:, price_col_idx]
    
    # Reconstruct Price_t = Price_{t-1} * exp(log_return)
    reconstructed_price = price_t_minus_1 * np.exp(unscaled_target)
    
    return reconstructed_price
