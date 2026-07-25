<div align="center">
  <h1>📈 AI-Powered Commodity Price Predictor</h1>
  <p><i>An end-to-end AI-driven price prediction system for Bitcoin, Gold and Silver using LSTM deep learning.</i></p>
</div>

<hr>

<h2>🎯 Project Overview</h2>
<p>
  Predicting financial markets is notoriously difficult due to noise and volatility. This system implements a <b>complete machine learning pipeline</b> — from data collection and feature engineering to LSTM-based forecasting and interactive dashboard visualization. It is designed as a decision-support tool to help users analyze historical price trends and view AI-generated price forecasts.
</p>

<hr>

<h2>🚀 Key Features</h2>

<ul>
  <li>
    <b>Multi-Asset Support:</b> 
    Unified prediction pipeline for Bitcoin (BTC-USD), Gold (GC=F) and Silver (SI=F) using Yahoo Finance data.
  </li>
  <li>
    <b>Deep Learning Forecasting:</b> 
    Optimized <code>LSTM</code> (Long Short-Term Memory) network with 100 units, 30-day lookback and 10% dropout for next-day price prediction.
  </li>
  <li>
    <b>Technical Indicator Suite:</b> 
    Automated calculation of 16 features including SMA, EMA, RSI, MACD and Bollinger Bands.
  </li>
  <li>
    <b>Baseline Comparisons:</b> 
    LSTM evaluated against Naive, Linear Regression, Random Forest, ARIMA and ARIMA-LSTM Ensemble baselines.
  </li>
  <li>
    <b>Interactive Dashboard:</b> 
    Streamlit-based web UI with Plotly charts, technical indicator overlays, performance metrics and AI prediction controls.
  </li>
  <li>
    <b>Live Data Sync:</b> 
    Integration with Yahoo Finance (yfinance) for up-to-date market data fetching.
  </li>
</ul>

<hr>

<h2>🏗️ Project Structure</h2>

<pre>
├── app/                  # Streamlit dashboard
├── src/
│   ├── data/             # Data collection, preprocessing, stationarity tests
│   ├── models/           # LSTM, ARIMA, baseline and ensemble models
│   ├── training/         # Final training scripts (BTC, Gold, Silver)
│   ├── evaluation/       # Walk-forward validation, backtesting, metrics
│   ├── experiments/      # Hyperparameter tuning (units, seq_len, dropout)
│   └── inference/        # Prediction pipeline
├── notebooks/            # Jupyter notebooks for exploration
├── data/                 # Raw, processed data and model artifacts
├── results/              # Plots, metrics CSVs and experiment logs
├── tests/                # Unit and pipeline tests
└── config.py             # Central configuration and hyperparameters
</pre>

<hr>

<h2>⚡ Quick Start</h2>

<pre>
pip install -r requirements.txt
streamlit run app/streamlit_app.py
</pre>

<hr>

<div align="center">
  <p>Built with ❤️ by <b>teamlocalhost</b></p>
  <p><i>University of Lahore — BSCS Fall 2022 to 2026</i></p>
</div>
