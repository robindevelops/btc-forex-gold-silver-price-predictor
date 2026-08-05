<div align="center">
  <h1>📈 AI-Powered Multi-Asset Price Predictor</h1>
  <p><i>An end-to-end ML price prediction system for Bitcoin, Gold and Silver using LSTM, GRU, LightGBM, CatBoost, and Stacked Ensemble models.</i></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.11-blue" alt="Python">
    <img src="https://img.shields.io/badge/TensorFlow-2.16-orange" alt="TensorFlow">
    <img src="https://img.shields.io/badge/Streamlit-1.50-red" alt="Streamlit">
    <img src="https://img.shields.io/badge/FastAPI-0.100-green" alt="FastAPI">
  </p>
</div>

<hr>

<h2>🎯 Project Overview</h2>
<p>
  Predicting financial markets is notoriously difficult due to noise and volatility. This system implements a <b>complete machine learning pipeline</b> — from data collection and feature engineering to multi-model forecasting and interactive dashboard visualization. It is designed as a decision-support tool to help users analyze historical price trends and view AI-generated price forecasts.
</p>

<hr>

<h2>🚀 Key Features</h2>

<ul>
  <li>
    <b>Multi-Asset Support:</b> 
    Unified prediction pipeline for Bitcoin (BTC-USD), Gold (GC=F) and Silver (SI=F) using Yahoo Finance data.
  </li>
  <li>
    <b>Multi-Model Architecture:</b> 
    <code>LSTM</code>, <code>GRU</code>, <code>LightGBM</code>, <code>CatBoost</code>, and <code>Stacked Ensemble</code> (Ridge meta-model) with walk-forward cross-validation.
  </li>
  <li>
    <b>30+ Technical Indicators:</b> 
    SMA, EMA, RSI, MACD, Bollinger Bands, ATR, VWAP, Stochastic Oscillator, Williams %R, ADX, CCI, ROC, lag returns, rolling volatility, calendar features.
  </li>
  <li>
    <b>External Macro Data:</b> 
    DXY (US Dollar Index), Crude Oil, S&P 500, VIX, Treasury Yields, and Bitcoin Fear & Greed Index.
  </li>
  <li>
    <b>Baseline Comparisons:</b> 
    Models evaluated against Naive, Linear Regression, Random Forest, XGBoost, ARIMA, and Gradient Boosting baselines.
  </li>
  <li>
    <b>Interactive Dashboard:</b> 
    Streamlit-based web UI with Plotly charts, technical indicator overlays, multi-model comparison, and AI prediction controls.
  </li>
  <li>
    <b>REST API:</b> 
    FastAPI endpoint for programmatic predictions (<code>GET /predict/{asset}</code>).
  </li>
  <li>
    <b>Production Ready:</b> 
    Docker support, CI/CD pipeline, automated retraining, centralized logging, and comprehensive test suite.
  </li>
</ul>

<hr>

<h2>🏗️ Project Structure</h2>

<pre>
├── app/                  # Streamlit dashboard
├── src/
│   ├── api/              # FastAPI REST endpoint
│   ├── data/             # Data collection, preprocessing, external data
│   ├── models/           # LSTM, GRU, LightGBM, CatBoost, ensemble
│   ├── training/         # Final training scripts (BTC, Gold, Silver)
│   ├── evaluation/       # Backtesting, cross-validation, metrics
│   ├── inference/        # Prediction pipeline
│   └── utils/            # Logging, metrics, reproducibility, transforms
├── tests/                # Unit and integration tests
├── scripts/              # Automation (retrain.py)
├── data/                 # Raw, processed data and model artifacts
├── notebooks/            # Jupyter notebooks for exploration
├── config.py             # Central configuration and hyperparameters
├── Dockerfile            # Docker containerization
├── docker-compose.yml    # Multi-service orchestration
├── Makefile              # Task automation
└── requirements.txt      # Pinned dependencies
</pre>

<hr>

<h2>⚡ Quick Start</h2>

<pre>
# 1. Clone and install
git clone &lt;repo-url&gt;
cd crypto-forex-prediction-system
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Collect data
python src/data/data_collection.py
python src/data/external_data.py

# 3. Preprocess features
python src/data/preprocessing.py

# 4. Train models (optional — pre-trained models included)
python src/training/train_final_btc.py
python src/training/train_final_gold.py
python src/training/train_final_silver.py

# 5. Launch dashboard
streamlit run app/streamlit_app.py

# 6. Or use the API
uvicorn src.api.app:app --reload
</pre>

<h3>Using Make (Recommended)</h3>
<pre>
make install      # Install dependencies
make collect-data # Fetch market data
make preprocess   # Feature engineering
make train        # Train all models
make serve        # Launch Streamlit dashboard
make test         # Run test suite
make all          # Full pipeline
</pre>

<h3>Using Docker</h3>
<pre>
docker-compose up --build
# Dashboard: http://localhost:8501
# API: http://localhost:8000
</pre>

<hr>

<h2>🧪 Testing</h2>
<pre>
python -m pytest tests/ -v
</pre>

<h2>📡 API Endpoints</h2>
<pre>
GET /health              # Health check
GET /predict/{asset}     # Predict next day price (Bitcoin, Gold, Silver)
GET /models              # List available models
</pre>

<hr>

<div align="center">
  <p>Built with ❤️ by <b>teamlocalhost</b></p>
  <p><i>University of Lahore — BSCS Fall 2022 to 2026</i></p>
</div>
