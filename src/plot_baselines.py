import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.linear_model import LinearRegression

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PROCESSED_DATA_DIR

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def flatten_sequences(X):
    return X.reshape(X.shape[0], -1)

def plot_best_baseline(asset_name):
    prefix = 'btc' if asset_name == 'Bitcoin' else asset_name.lower()

    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_test.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_test.npy"))

    X_train_flat = flatten_sequences(X_train)
    X_test_flat  = flatten_sequences(X_test)

    # Train the best baseline (Linear Regression)
    model = LinearRegression()
    model.fit(X_train_flat, y_train.ravel())
    y_pred = model.predict(X_test_flat)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label=f'Actual {asset_name} Price (Scaled)', color='#111827', linewidth=2)
    plt.plot(y_pred, label=f'Predicted (Linear Regression)', color='#3b82f6', linestyle='--', linewidth=2)
    
    plt.title(f'{asset_name} Price Prediction - Baseline Linear Regression', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Test Set Timesteps (Days)', fontsize=11)
    plt.ylabel('Scaled Price (0 to 1)', fontsize=11)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Clean up borders
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Save plot
    filepath = os.path.join(PLOTS_DIR, f'{prefix}_baseline_lr.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved plot for {asset_name} to {filepath}")

if __name__ == "__main__":
    for asset in ['Bitcoin', 'Gold', 'Silver']:
        plot_best_baseline(asset)
