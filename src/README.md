# Source Code Organization (`src/`)

This directory contains the entire machine learning pipeline for the Crypto & Forex Prediction System. It has been modularized into distinct subfolders based on the phase of the pipeline.

## 📂 Directory Structure

### `data/`
Scripts responsible for data acquisition, cleaning, preprocessing, and exploratory testing.
* **`data_collection.py`**: Fetches raw data from Yahoo Finance API.
* **`preprocessing.py`**: Handles missing values, scaling (MinMaxScaler), technical indicators (SMA/EMA), and sequence generation for the LSTM.
* **`stationarity_tests.py`**: Augmented Dickey-Fuller tests to analyze time-series stationarity.

### `models/`
Defines the core machine learning models and architectures.
* **`model_lstm.py`**: Neural network architecture building blocks (Keras).
* **`baseline_models.py`**: Traditional ML baselines (Linear Regression, Random Forest, XGBoost).
* **`arima_model.py`**: Autoregressive Integrated Moving Average pipeline for linear trend forecasting.
* **`ensemble_model.py`**: Hybrid approach combining ARIMA and LSTM predictions.

### `training/`
Scripts to execute the formal training pipeline for each asset.
* **`train_final_btc.py`**: 2-phase unbiased training pipeline for Bitcoin.
* **`train_final_gold.py`**: 2-phase unbiased training pipeline for Gold.
* **`train_final_silver.py`**: 2-phase unbiased training pipeline for Silver.
* **`training.py`**: Initial/legacy training loop implementations.

### `experiments/`
Isolated scripts used during the hyperparameter tuning phase.
* **`experiment_dropout.py`**: Tests varying dropout rates for regularization.
* **`experiment_lstm_units.py`**: Tests model capacity (e.g., 50 vs 100 hidden units).
* **`experiment_seq_length.py`**: Tests historical lookback windows (30, 60, 90 days).

### `evaluation/`
Scripts to evaluate model performance, validate generalizability, and generate metrics.
* **`evaluation.py`**: Unified module for calculating RMSE, MAE, MAPE, and R² for any model.
* **`walk_forward.py`**: Simulates production behavior by rolling a fixed window forward day-by-day.
* **`backtesting.py`**: Extensive final evaluation against naive models and buy-and-hold strategies.
* **`plot_baselines.py`**: Visualizations for initial baseline comparisons.

### `inference/`
Production-ready code for making future predictions.
* **`prediction.py`**: End-to-end pipeline to load a trained model, grab the latest live data, preprocess it, and output tomorrow's forecast.
