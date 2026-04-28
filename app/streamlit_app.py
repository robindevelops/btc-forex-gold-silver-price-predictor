import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

# Add the project root to the python path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import PROCESSED_DATA_DIR, MODELS_DIR, BEST_LSTM_CONFIG
from src.data.preprocessing import create_sequences
from src.data.sync_live_data import update_live_data

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

# ═══════════════════════════════════════════════════════════
#  PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Crypto & Forex AI Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium look
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    h1 {
        font-family: 'Inter', sans-serif;
        color: #FFFFFF;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  UTILITIES & CACHED DATA LOADING
# ═══════════════════════════════════════════════════════════

def get_prefix(asset):
    return 'btc' if asset == 'Bitcoin' else asset.lower()

@st.cache_data
def load_historical_data(asset):
    prefix = get_prefix(asset)
    path = os.path.join(PROCESSED_DATA_DIR, f'{prefix}_features.csv')
    try:
        if not os.path.exists(path):
            st.warning(f"Historical data not found for {asset}.")
            return None
        df = pd.read_csv(path, parse_dates=['timestamp'])
        df = df.sort_values('timestamp')
        return df
    except Exception as e:
        st.error(f"Failed to load historical data for {asset}: {str(e)}")
        return None

@st.cache_resource
def load_ai_model(asset):
    prefix = get_prefix(asset)
    model_path = os.path.join(MODELS_DIR, f'{prefix}_lstm_final.keras')
    scaler_path = os.path.join(MODELS_DIR, f'{prefix}_scaler.pkl')
    try:
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.error(f"Model or scaler not found for {asset}. Please ensure training has been completed.")
            return None, None
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading AI model for {asset}: {str(e)}")
        return None, None

@st.cache_data
def load_performance_metrics():
    path = os.path.join(RESULTS_DIR, 'final_performance_table.csv')
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
        st.warning("Performance metrics not found. Please run backtesting scripts.")
        return None
    except Exception as e:
        st.error(f"Error loading performance metrics: {str(e)}")
        return None

@st.cache_data
def load_scaled_data(asset):
    """Loads the scaled data for inference, prioritizing live data if available."""
    prefix = get_prefix(asset)
    try:
        live_path = os.path.join(PROCESSED_DATA_DIR, f'{prefix}_live_scaled.csv')
        if os.path.exists(live_path):
            return pd.read_csv(live_path, index_col='timestamp', parse_dates=True)
            
        # Fallback to concatenated historical splits
        train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_train_scaled.csv'), index_col='timestamp', parse_dates=True)
        val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_val_scaled.csv'), index_col='timestamp', parse_dates=True)
        test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_test_scaled.csv'), index_col='timestamp', parse_dates=True)
        return pd.concat([train_df, val_df, test_df])
    except Exception as e:
        st.error(f"Failed to load scaled datasets for {asset}: {str(e)}")
        return None

def inverse_transform_price(scaled_values, scaler, price_col_idx=0, n_features=16):
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, price_col_idx] = np.array(scaled_values).ravel()
    return scaler.inverse_transform(dummy)[:, price_col_idx]

# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
st.sidebar.title("AI Prediction Engine")
st.sidebar.markdown("Select an asset below to analyze historical trends and generate deep-learning price forecasts.")

asset_selection = st.sidebar.selectbox(
    "Target Asset",
    options=["Bitcoin", "Gold", "Silver"],
    index=0
)

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Sync Live Market Data", use_container_width=True):
    with st.spinner(f"Fetching latest data from Yahoo Finance for {asset_selection}..."):
        success = update_live_data(asset_selection)
        if success:
            st.cache_data.clear()
            st.sidebar.success("Market Data Synced!")
            st.rerun()
        else:
            st.sidebar.error("Failed to sync live data.")

st.sidebar.markdown("---")

# Technical Indicator Toggles
st.sidebar.subheader("Technical Indicators")
show_bb = st.sidebar.checkbox("Bollinger Bands")
show_rsi = st.sidebar.checkbox("RSI (14)")
show_macd = st.sidebar.checkbox("MACD (12, 26, 9)")

st.sidebar.markdown("---")

if st.sidebar.button("🚀 Run AI Prediction", use_container_width=True):
    st.session_state['run_prediction'] = True
else:
    if 'run_prediction' not in st.session_state:
        st.session_state['run_prediction'] = False

st.sidebar.markdown("---")
st.sidebar.info("Model: **LSTM (Optimized)**\n\nFeatures: **Price, Vol, SMA, EMA, RSI, MACD**\n\nLookback: **30 Days**")

# Define colors based on asset
color_map = {
    "Bitcoin": "#F7931A", # Bitcoin Orange
    "Gold": "#FFD700",    # Gold
    "Silver": "#C0C0C0"   # Silver
}

# ═══════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════
st.title(f"📈 {asset_selection} Analysis Dashboard")

tab1, tab2 = st.tabs(["📊 Price Forecast & Indicators", "⚙️ Performance Metrics"])

# Load data safely
df = load_historical_data(asset_selection)

with tab1:
    if df is not None:
        # Display top-level metrics
        latest_date = df['timestamp'].iloc[-1].strftime("%Y-%m-%d")
        latest_price = df['price'].iloc[-1]
        prev_price = df['price'].iloc[-2]
        daily_change = latest_price - prev_price
        pct_change = (daily_change / prev_price) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label=f"Latest Close ({latest_date})", 
                      value=f"${latest_price:,.2f}", 
                      delta=f"{daily_change:+,.2f} ({pct_change:+.2f}%)")
        with col2:
            st.metric(label="30-Day High", value=f"${df['price'].tail(30).max():,.2f}")
        with col3:
            st.metric(label="30-Day Low", value=f"${df['price'].tail(30).min():,.2f}")

        st.markdown("---")
        
        # ── Interactive Plotly Chart with Indicators ──
        st.subheader(f"Historical {asset_selection} Price & Technical Analysis")
        
        try:
            # Determine subplot rows based on selected indicators
            n_rows = 1
            row_heights = [1.0]
            
            if show_rsi or show_macd:
                row_heights = [0.6]
            if show_rsi:
                n_rows += 1
                row_heights.append(0.2)
            if show_macd:
                n_rows += 1
                row_heights.append(0.2)
                
            fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=row_heights)
            
            current_row = 1
            
            # Convert hex to rgba for the fill color
            hex_color = color_map[asset_selection]
            rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            fill_rgba = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.1)"
            
            # 1. Main Price Chart
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'], y=df['price'], mode='lines', name=f'{asset_selection} Price',
                    line=dict(color=color_map[asset_selection], width=2),
                    fill='tozeroy', fillcolor=fill_rgba
                ), row=current_row, col=1
            )
            
            # Overlay Bollinger Bands if selected
            if show_bb and 'BB_Upper' in df.columns:
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['BB_Upper'], mode='lines', name='BB Upper',
                                         line=dict(color='gray', width=1, dash='dash')), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['BB_Lower'], mode='lines', name='BB Lower',
                                         line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['BB_Mid'], mode='lines', name='BB Mid (SMA 20)',
                                         line=dict(color='rgba(255,255,255,0.3)', width=1)), row=current_row, col=1)

            # 2. RSI Subplot
            if show_rsi and 'RSI' in df.columns:
                current_row += 1
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], mode='lines', name='RSI (14)',
                                         line=dict(color='#E91E63', width=1.5)), row=current_row, col=1)
                # Overbought / Oversold lines
                fig.add_hline(y=70, line=dict(color='red', width=1, dash='dot'), row=current_row, col=1)
                fig.add_hline(y=30, line=dict(color='green', width=1, dash='dot'), row=current_row, col=1)
                fig.update_yaxes(title_text="RSI", row=current_row, col=1, range=[0, 100])

            # 3. MACD Subplot
            if show_macd and 'MACD' in df.columns:
                current_row += 1
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MACD'], mode='lines', name='MACD',
                                         line=dict(color='#2196F3', width=1.5)), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MACD_Signal'], mode='lines', name='Signal',
                                         line=dict(color='#FF9800', width=1.5)), row=current_row, col=1)
                
                # MACD Histogram
                macd_hist = df['MACD'] - df['MACD_Signal']
                colors = ['#4CAF50' if val >= 0 else '#F44336' for val in macd_hist]
                fig.add_trace(go.Bar(x=df['timestamp'], y=macd_hist, name='Histogram',
                                     marker_color=colors), row=current_row, col=1)
                fig.update_yaxes(title_text="MACD", row=current_row, col=1)

            # Update layout
            total_height = 500 if n_rows == 1 else 700
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=total_height,
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Add y-axis label to the main chart
            fig.update_yaxes(title_text="Price (USD)", tickprefix="$", row=1, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering chart: {str(e)}")

        # ═══════════════════════════════════════════════════════════
        #  AI PREDICTION ENGINE
        # ═══════════════════════════════════════════════════════════
        if st.session_state['run_prediction']:
            st.markdown("---")
            st.header("🤖 AI Prediction Engine")
            
            with st.spinner(f"Loading {asset_selection} neural network and running inference..."):
                try:
                    model, scaler = load_ai_model(asset_selection)
                    if model is None or scaler is None:
                        st.stop()
                    
                    full_df = load_scaled_data(asset_selection)
                    if full_df is None:
                        st.stop()
                        
                    seq_len = BEST_LSTM_CONFIG['seq_len']
                    n_features = full_df.shape[1]
                    price_col_idx = list(full_df.columns).index('price')
                    
                    # 1. Walk-Forward Visual (Last 150 days)
                    recent_df = full_df.tail(150 + seq_len)
                    X_recent, y_recent = create_sequences(recent_df, seq_len=seq_len)
                    
                    y_pred_scaled = model.predict(X_recent, verbose=0)
                    y_recent_usd = inverse_transform_price(y_recent, scaler, price_col_idx, n_features)
                    y_pred_usd = inverse_transform_price(y_pred_scaled, scaler, price_col_idx, n_features)
                    recent_dates = recent_df.index[seq_len:]
                    
                    # 2. Predict Tomorrow (Inference)
                    last_sequence = full_df.values[-seq_len:].reshape(1, seq_len, n_features)
                    tomorrow_scaled = model.predict(last_sequence, verbose=0)
                    tomorrow_usd = inverse_transform_price(tomorrow_scaled, scaler, price_col_idx, n_features)[0]
                    
                    # Display Next Day Prediction
                    predicted_change = tomorrow_usd - latest_price
                    pct_pred_change = (predicted_change / latest_price) * 100
                    
                    st.subheader("Tomorrow's Forecast")
                    
                    st.info(f"The Deep Learning model predicts the price of **{asset_selection}** will move to **${tomorrow_usd:,.2f}** on the next trading day.")
                    
                    st.metric(
                        label=f"Predicted Close for Next Trading Day", 
                        value=f"${tomorrow_usd:,.2f}", 
                        delta=f"{predicted_change:+,.2f} ({pct_pred_change:+.2f}%) projected"
                    )
                    
                    st.markdown("---")
                    
                    # Plot Actual vs Predicted on Recent Period
                    st.subheader("Model Validation: Recent Market Performance")
                    st.markdown("This chart compares the model's Walk-Forward predictions against the actual prices over the last 150 days to visualize current momentum tracking.")
                    
                    fig_pred = go.Figure()
                    
                    # Actual Test Prices
                    fig_pred.add_trace(go.Scatter(
                        x=recent_dates, y=y_recent_usd, mode='lines', name='Actual Price',
                        line=dict(color=color_map[asset_selection], width=2)
                    ))
                    
                    # Predicted Test Prices
                    fig_pred.add_trace(go.Scatter(
                        x=recent_dates, y=y_pred_usd, mode='lines', name='LSTM Predicted',
                        line=dict(color='#00ffcc', width=2, dash='dash')
                    ))
                    
                    fig_pred.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(title="Date", showgrid=False),
                        yaxis=dict(title="Price (USD)", showgrid=True, gridcolor="rgba(255,255,255,0.1)", tickprefix="$"),
                        height=400,
                        margin=dict(l=0, r=0, t=30, b=0),
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)

                except Exception as e:
                    st.error(f"An error occurred during AI Inference: {str(e)}")

with tab2:
    st.header("⚙️ Model Performance Metrics")
    perf_df = load_performance_metrics()
    
    if perf_df is not None:
        try:
            asset_metrics = perf_df[perf_df['Asset'] == asset_selection]
            
            # Check if the LSTM row exists
            lstm_rows = asset_metrics[asset_metrics['Model'] == 'LSTM (Optimized)']
            if not lstm_rows.empty:
                lstm_row = lstm_rows.iloc[0]
                
                st.subheader("LSTM Error Metrics (Test Set)")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("RMSE", f"${lstm_row['RMSE']:,.2f}")
                with m2:
                    st.metric("MAE", f"${lstm_row['MAE']:,.2f}")
                with m3:
                    st.metric("MAPE", f"{lstm_row['MAPE']:.2f}%")
                with m4:
                    st.metric("R² Score", f"{lstm_row['R2']:.4f}")
                    
                st.markdown("---")
                st.subheader("Baseline Comparison (RMSE)")
                st.markdown("Comparing our optimized LSTM against traditional machine learning baselines. **Lower RMSE is better.**")
                
                # Prepare chart dataframe
                chart_df = asset_metrics.sort_values('RMSE', ascending=False)
                
                # Assign colors
                colors = []
                for m in chart_df['Model']:
                    if m == 'LSTM (Optimized)':
                        colors.append('#00ffcc') # Highlight LSTM
                    elif 'Naive' in m:
                        colors.append('#FF5722') # Highlight Naive
                    else:
                        colors.append('#607D8B') # Gray for others
                
                fig_bar = go.Figure(go.Bar(
                    x=chart_df['RMSE'],
                    y=chart_df['Model'],
                    orientation='h',
                    marker=dict(color=colors)
                ))
                
                fig_bar.update_layout(
                    template='plotly_dark',
                    xaxis_title="Root Mean Squared Error (USD)",
                    yaxis_title="",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                fig_bar.update_traces(texttemplate='$%{x:,.0f}', textposition='outside')
                
                st.plotly_chart(fig_bar, use_container_width=True)
                
                st.info("💡 **Note on Naive Baseline:** In financial time series, the 'Naive' forecast (guessing tomorrow's price will exactly match today's price) often produces the lowest raw error because daily price movements are small. Our goal is to build the best *intelligent* model that successfully learns patterns, where our LSTM significantly outperforms traditional Linear Regression and Random Forest.")
                
            else:
                st.warning("LSTM metrics not found for this asset in the performance table.")
        except Exception as e:
            st.error(f"Error rendering performance metrics: {str(e)}")
