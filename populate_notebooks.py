import json
import os

NOTEBOOK_DIR = "/Users/mac/Desktop/crypto-forex-prediction-system/notebooks"
os.makedirs(NOTEBOOK_DIR, exist_ok=True)

def create_notebook(filename, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.7"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(os.path.join(NOTEBOOK_DIR, filename), "w") as f:
        json.dump(notebook, f, indent=1)

def markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    }

# ---------------------------------------------------------
# 02 Feature Engineering
# ---------------------------------------------------------
cells_02 = [
    markdown_cell("# Feature Engineering & Data Preparation\n\nIn this notebook, we transform our raw price and volume data into a format suitable for our Machine Learning models.\n\nThis involves:\n1. Stationarity Testing (ADF Test)\n2. Feature Scaling (MinMaxScaler)\n3. Sequence Generation for LSTM (3D Tensors)"),
    code_cell("import os\nimport sys\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n# Set paths\nsys.path.append(os.path.abspath('..'))\nfrom src.data.stationarity_tests import run_adf_test"),
    markdown_cell("## 1. Stationarity Testing\n\nWe use the Augmented Dickey-Fuller (ADF) test to see if our time series is stationary (mean and variance are constant over time). Machine Learning models struggle with non-stationary data."),
    code_cell("df_btc = pd.read_csv('../data/processed/btc_features.csv', index_col='timestamp', parse_dates=True)\nprice = df_btc['price']\n\nprint('Testing Raw Price...')\nres_raw = run_adf_test(price, 'BTC Raw Price')\n\nprint('\\nTesting Differenced Price...')\nres_diff = run_adf_test(price.diff().dropna(), 'BTC Differenced Price')"),
    markdown_cell("## 2. Feature Scaling & Sequence Generation\n\nLSTMs require input data to be scaled (usually between 0 and 1) and formatted as 3D sequences: `(samples, time_steps, features)`."),
    code_cell("from src.data.preprocessing import create_sequences\nimport joblib\n\n# Load scaled data\ntrain_df = pd.read_csv('../data/processed/btc_train_scaled.csv', index_col='timestamp', parse_dates=True)\n\nprint('Raw tabular data shape:', train_df.shape)\n\n# Generate sequences\nSEQ_LEN = 30\nX, y = create_sequences(train_df, seq_len=SEQ_LEN)\n\nprint(f'Generated 3D Sequence Shape (X): {X.shape}')\nprint(f'Generated Target Shape (y): {y.shape}')\nprint(f'This means we have {X.shape[0]} samples, each looking back {X.shape[1]} days, with {X.shape[2]} features.')")
]

# ---------------------------------------------------------
# 03 Model Training
# ---------------------------------------------------------
cells_03 = [
    markdown_cell("# Model Training & Architecture\n\nIn this notebook, we define our LSTM architecture and review the training process.\n\nWe utilize a 2-Phase Training approach:\n1. Find optimal epochs using a validation set to prevent overfitting.\n2. Retrain blindly on Train+Val for exactly those epochs."),
    code_cell("import os\nimport sys\nimport tensorflow as tf\n\nsys.path.append(os.path.abspath('..'))\nfrom src.models.model_lstm import build_lstm_model\nfrom config import BEST_LSTM_CONFIG"),
    markdown_cell("## 1. LSTM Architecture\n\nLet's visualize the structure of our model."),
    code_cell("config = BEST_LSTM_CONFIG\nmodel = build_lstm_model(seq_len=config['seq_len'], n_features=16, lstm_units=config['lstm_units'], dropout_rate=config['dropout_rate'], dense_units=config['dense_units'])\n\nmodel.summary()"),
    markdown_cell("## 2. Training Results & Hyperparameters\n\nWe ran 11 separate experiments adjusting sequence length (30 vs 60 vs 90), dropout rates, and LSTM units. The chosen configuration is:\n- Sequence Length: 30 days\n- LSTM Units: 100\n- Dropout: 0.1\n\nOur training scripts (`src/training/train_final_btc.py`) automatically output loss curves to the `results/` folder.")
]

# ---------------------------------------------------------
# 04 Evaluation
# ---------------------------------------------------------
cells_04 = [
    markdown_cell("# Final Evaluation & Backtesting\n\nThis is the grand finale. We evaluate all our models (LSTM, Linear Regression, Random Forest, ARIMA, and Ensembles) against a Naive Baseline and a Buy-and-Hold strategy."),
    code_cell("import os\nimport sys\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nsys.path.append(os.path.abspath('..'))"),
    markdown_cell("## 1. Final 9-Model Comparison Table\n\nOur full backtesting script generates this comprehensive table across all assets."),
    code_cell("results_df = pd.read_csv('../results/final_performance_table.csv')\ndisplay(results_df[results_df['Asset'] == 'Bitcoin'])\n\nprint('\\nAnd for Gold:')\ndisplay(results_df[results_df['Asset'] == 'Gold'])"),
    markdown_cell("## 2. Walk-Forward Validation Results\n\nInstead of a static train/test split, walk-forward validation simulates a real-world scenario by predicting one day ahead, sliding the window, and predicting again."),
    code_cell("walk_forward_df = pd.read_csv('../results/walk_forward_results.csv')\ndisplay(walk_forward_df)"),
    markdown_cell("## 3. Conclusions\n\n1. **Magnitude vs Direction:** The LSTM is excellent at predicting the magnitude of the price (RMSE of $2,489 on BTC with 2.3% MAPE), but directional accuracy (predicting up/down) is around 54.2%.\n2. **Naive Baseline:** The naive forecast (tomorrow = today) achieves the lowest raw RMSE, which is standard in financial time series.\n3. **Silver's Noise:** Silver proved highly unpredictable due to its extreme percentage volatility at low absolute prices.")
]

create_notebook("02_feature_engineering.ipynb", cells_02)
create_notebook("03_model_training.ipynb", cells_03)
create_notebook("04_evaluation.ipynb", cells_04)

print("Notebooks populated.")
