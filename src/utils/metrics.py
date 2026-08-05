import numpy as np

def compute_rmse(y_true, y_pred) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

def compute_mae(y_true, y_pred) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def compute_mape(y_true, y_pred) -> float:
    """Compute Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def compute_r2(y_true, y_pred) -> float:
    """Compute R-squared (Coefficient of Determination)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - (ss_res / ss_tot))

def compute_directional_accuracy(y_true, y_pred, y_prev) -> float:
    """Compute Directional Accuracy."""
    y_true, y_pred, y_prev = np.array(y_true), np.array(y_pred), np.array(y_prev)
    true_direction = np.sign(y_true - y_prev)
    pred_direction = np.sign(y_pred - y_prev)
    
    # Ignore zero movements if necessary, but generally they count as match if both zero
    correct = (true_direction == pred_direction)
    return float(np.mean(correct) * 100)

def compute_all_metrics(y_true, y_pred, y_prev=None) -> dict:
    """
    Compute all available metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_prev: Previous values (optional, required for directional accuracy)
        
    Returns:
        dict: Dictionary containing all computed metrics
    """
    metrics = {
        'rmse': compute_rmse(y_true, y_pred),
        'mae': compute_mae(y_true, y_pred),
        'mape': compute_mape(y_true, y_pred),
        'r2': compute_r2(y_true, y_pred)
    }
    
    if y_prev is not None:
        metrics['directional_accuracy'] = compute_directional_accuracy(y_true, y_pred, y_prev)
        
    return metrics
