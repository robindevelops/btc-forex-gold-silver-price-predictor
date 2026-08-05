"""
Streamlit Dashboard — Crypto & Forex AI Price Predictor.

Multi-asset (Bitcoin, Gold, Silver) price analysis and AI-powered
forecasting dashboard with technical indicator overlays, model
performance comparison, and next-day prediction.
"""

import os
import sys
import logging
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add the project root to the python path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import PROCESSED_DATA_DIR, MODELS_DIR, BEST_LSTM_CONFIG, MODEL_STATUS
from src.data.preprocessing import create_sequences
from src.data.sync_live_data import update_live_data
from src.utils.inverse_transform import reconstruct_price

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'results'))

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp {
        background-color: #0E1117;
    }
    h1, h2, h3 {
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
        font-size: 2.2rem;
    }
    .status-active {
        color: #4CAF50;
        font-weight: 600;
    }
    .status-disabled {
        color: #F44336;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  UTILITIES & CACHED DATA LOADING
# ═══════════════════════════════════════════════════════════

def get_prefix(asset: str) -> str:
    """Get file prefix for an asset name."""
    return 'btc' if asset == 'Bitcoin' else asset.lower()

@st.cache_data(ttl=3600)
def load_historical_data(asset: str) -> Optional[pd.DataFrame]:
    """Load unscaled feature data for charting."""
    prefix = get_prefix(asset)
    path = os.path.join(PROCESSED_DATA_DIR, f'{prefix}_features.csv')
    try:
        if not os.path.exists(path):
            st.warning(f"Historical data not found for {asset}. Run preprocessing first.")
            return None
        df = pd.read_csv(path, parse_dates=['timestamp'])
        df = df.sort_values('timestamp')
        return df
    except Exception as e:
        st.error(f"Failed to load historical data for {asset}: {e}")
        return None

@st.cache_resource
def load_keras_model(model_path: str):
    """Load a Keras model (.keras or .h5) with caching."""
    try:
        from tensorflow.keras.models import load_model
        if not os.path.exists(model_path):
            return None
        return load_model(model_path)
    except Exception as e:
        logger.error(f"Error loading Keras model {model_path}: {e}")
        return None

@st.cache_resource
def load_scaler(asset: str):
    """Load the fitted scaler for an asset."""
    prefix = get_prefix(asset)
    scaler_path = os.path.join(MODELS_DIR, f'{prefix}_scaler.pkl')
    try:
        if not os.path.exists(scaler_path):
            return None
        return joblib.load(scaler_path)
    except Exception as e:
        logger.error(f"Error loading scaler for {asset}: {e}")
        return None

@st.cache_data(ttl=3600)
def load_performance_metrics() -> Optional[pd.DataFrame]:
    """Load saved performance metrics table."""
    path = os.path.join(RESULTS_DIR, 'final_performance_table.csv')
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
        return None
    except Exception as e:
        logger.error(f"Error loading performance metrics: {e}")
        return None

@st.cache_data(ttl=3600)
def load_scaled_data(asset: str) -> Optional[pd.DataFrame]:
    """Load scaled data for inference, prioritizing live data if available."""
    prefix = get_prefix(asset)
    try:
        live_path = os.path.join(PROCESSED_DATA_DIR, f'{prefix}_live_scaled.csv')
        if os.path.exists(live_path):
            return pd.read_csv(live_path, index_col='timestamp', parse_dates=True)

        # Fallback to concatenated historical splits
        paths = [
            os.path.join(PROCESSED_DATA_DIR, f'{prefix}_train_scaled.csv'),
            os.path.join(PROCESSED_DATA_DIR, f'{prefix}_val_scaled.csv'),
            os.path.join(PROCESSED_DATA_DIR, f'{prefix}_test_scaled.csv'),
        ]
        if all(os.path.exists(p) for p in paths):
            dfs = [pd.read_csv(p, index_col='timestamp', parse_dates=True) for p in paths]
            return pd.concat(dfs)
        return None
    except Exception as e:
        st.error(f"Failed to load scaled datasets for {asset}: {e}")
        return None


def get_available_models(asset: str) -> dict:
    """Check which models are available on disk for an asset."""
    prefix = get_prefix(asset)
    models = {}

    # LSTM
    lstm_path = os.path.join(MODELS_DIR, f'{prefix}_lstm_final.keras')
    if os.path.exists(lstm_path):
        models['LSTM'] = lstm_path

    # GRU
    gru_path = os.path.join(MODELS_DIR, f'{prefix}_gru_final.keras')
    if os.path.exists(gru_path):
        models['GRU'] = gru_path

    # LightGBM
    lgbm_path = os.path.join(MODELS_DIR, f'{prefix}_lgbm_final.pkl')
    if os.path.exists(lgbm_path):
        models['LightGBM'] = lgbm_path

    # CatBoost
    cb_path = os.path.join(MODELS_DIR, f'{prefix}_catboost_final.cbm')
    if os.path.exists(cb_path):
        models['CatBoost'] = cb_path

    # Meta-model (Ensemble)
    meta_path = os.path.join(MODELS_DIR, f'{prefix}_metamodel.pkl')
    if not os.path.exists(meta_path):
        # Check alternate naming
        meta_path = os.path.join(MODELS_DIR, f'{prefix}_meta_model.pkl')
    if os.path.exists(meta_path):
        models['Ensemble'] = meta_path

    return models


def run_model_prediction(asset: str, model_name: str, full_df: pd.DataFrame, scaler) -> Optional[float]:
    """Run prediction for a specific model and return predicted scaled value."""
    prefix = get_prefix(asset)
    seq_len = BEST_LSTM_CONFIG['seq_len']
    n_features = full_df.shape[1]

    try:
        if model_name in ('LSTM', 'GRU'):
            model_path = os.path.join(MODELS_DIR, f'{prefix}_{model_name.lower()}_final.keras')
            model = load_keras_model(model_path)
            if model is None:
                return None
            last_seq = full_df.values[-seq_len:].reshape(1, seq_len, n_features)
            pred = model.predict(last_seq, verbose=0)[0, 0]
            return pred

        elif model_name == 'LightGBM':
            lgbm_path = os.path.join(MODELS_DIR, f'{prefix}_lgbm_final.pkl')
            if not os.path.exists(lgbm_path):
                return None
            lgbm = joblib.load(lgbm_path)
            last_seq = full_df.values[-seq_len:].reshape(1, -1)
            pred = lgbm.predict(last_seq)[0]
            return pred

        elif model_name == 'CatBoost':
            cb_path = os.path.join(MODELS_DIR, f'{prefix}_catboost_final.cbm')
            if not os.path.exists(cb_path):
                return None
            from catboost import CatBoostRegressor
            cb = CatBoostRegressor()
            cb.load_model(cb_path)
            last_features = full_df.values[-1:, :]
            pred = cb.predict(last_features)[0]
            return pred

    except Exception as e:
        logger.error(f"Prediction error for {model_name}/{asset}: {e}")
        return None

    return None


# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
st.sidebar.title("AI Prediction Engine")
st.sidebar.markdown("Select an asset to analyze historical trends and generate AI-powered price forecasts.")

asset_selection = st.sidebar.selectbox(
    "Target Asset",
    options=["Bitcoin", "Gold", "Silver"],
    index=0
)

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Sync Live Market Data", use_container_width=True):
    with st.spinner(f"Fetching latest data from Yahoo Finance for {asset_selection}..."):
        try:
            success = update_live_data(asset_selection)
            if success:
                st.cache_data.clear()
                st.sidebar.success("Market Data Synced!")
                st.rerun()
            else:
                st.sidebar.error("Failed to sync live data. Check logs.")
        except Exception as e:
            st.sidebar.error(f"Sync error: {e}")

st.sidebar.markdown("---")

# Technical Indicator Toggles
st.sidebar.subheader("Technical Indicators")
show_bb = st.sidebar.checkbox("Bollinger Bands")
show_rsi = st.sidebar.checkbox("RSI (14)")
show_macd = st.sidebar.checkbox("MACD (12, 26, 9)")
show_volume = st.sidebar.checkbox("Volume", value=True)

st.sidebar.markdown("---")

# Model selection
available_models = get_available_models(asset_selection)
model_status = MODEL_STATUS.get(asset_selection, {})
primary_model = model_status.get('primary_model', 'LSTM')

st.sidebar.subheader("AI Model")
model_options = list(available_models.keys()) if available_models else ["LSTM"]
selected_model = st.sidebar.selectbox(
    "Select Model",
    options=model_options,
    index=0
)

if st.sidebar.button("🚀 Run AI Prediction", use_container_width=True):
    st.session_state['run_prediction'] = True
else:
    if 'run_prediction' not in st.session_state:
        st.session_state['run_prediction'] = False

st.sidebar.markdown("---")

# Model info
st.sidebar.info(
    f"**Primary Model:** {primary_model}\n\n"
    f"**Available Models:** {', '.join(available_models.keys()) if available_models else 'None'}\n\n"
    f"**Lookback:** {BEST_LSTM_CONFIG['seq_len']} Days\n\n"
    f"**Status:** {model_status.get('status', 'unknown')}"
)

# Define colors based on asset
color_map = {
    "Bitcoin": "#F7931A",
    "Gold": "#FFD700",
    "Silver": "#C0C0C0"
}

# ═══════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════
st.title(f"📈 {asset_selection} Analysis Dashboard")

tab1, tab2, tab3 = st.tabs([
    "📊 Price Forecast & Indicators",
    "⚙️ Performance Metrics",
    "📋 Model Inventory"
])

# Load data safely
df = load_historical_data(asset_selection)

with tab1:
    if df is not None and len(df) > 1:
        # Display top-level metrics
        latest_date = df['timestamp'].iloc[-1].strftime("%Y-%m-%d")
        latest_price = df['price'].iloc[-1]
        prev_price = df['price'].iloc[-2]
        daily_change = latest_price - prev_price
        pct_change = (daily_change / prev_price) * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label=f"Latest Close ({latest_date})",
                value=f"${latest_price:,.2f}",
                delta=f"{daily_change:+,.2f} ({pct_change:+.2f}%)"
            )
        with col2:
            st.metric(label="30-Day High", value=f"${df['price'].tail(30).max():,.2f}")
        with col3:
            st.metric(label="30-Day Low", value=f"${df['price'].tail(30).min():,.2f}")
        with col4:
            if 'volume' in df.columns:
                avg_vol = df['volume'].tail(30).mean()
                st.metric(label="Avg 30D Volume", value=f"{avg_vol:,.0f}")

        st.markdown("---")

        # ── Interactive Plotly Chart with Indicators ──
        st.subheader(f"Historical {asset_selection} Price & Technical Analysis")

        try:
            # Determine subplot rows based on selected indicators
            n_rows = 1
            row_heights = [1.0]

            if show_volume or show_rsi or show_macd:
                row_heights = [0.5]
            if show_volume:
                n_rows += 1
                row_heights.append(0.15)
            if show_rsi:
                n_rows += 1
                row_heights.append(0.15)
            if show_macd:
                n_rows += 1
                row_heights.append(0.2)

            fig = make_subplots(
                rows=n_rows, cols=1, shared_xaxes=True,
                vertical_spacing=0.04, row_heights=row_heights
            )

            current_row = 1

            # Convert hex to rgba for the fill color
            hex_color = color_map[asset_selection]
            rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            fill_rgba = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.1)"

            # 1. Main Price Chart
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'], y=df['price'], mode='lines',
                    name=f'{asset_selection} Price',
                    line=dict(color=color_map[asset_selection], width=2),
                    fill='tozeroy', fillcolor=fill_rgba
                ), row=current_row, col=1
            )

            # Overlay Bollinger Bands
            if show_bb and 'BB_Upper' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['BB_Upper'], mode='lines',
                    name='BB Upper', line=dict(color='gray', width=1, dash='dash')
                ), row=current_row, col=1)
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['BB_Lower'], mode='lines',
                    name='BB Lower', line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
                ), row=current_row, col=1)
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['BB_Mid'], mode='lines',
                    name='BB Mid (SMA 20)',
                    line=dict(color='rgba(255,255,255,0.3)', width=1)
                ), row=current_row, col=1)

            # 2. Volume Subplot
            if show_volume and 'volume' in df.columns:
                current_row += 1
                vol_colors = ['#4CAF50' if df['price'].iloc[i] >= df['price'].iloc[max(0, i-1)]
                              else '#F44336' for i in range(len(df))]
                fig.add_trace(go.Bar(
                    x=df['timestamp'], y=df['volume'], name='Volume',
                    marker_color=vol_colors, opacity=0.6
                ), row=current_row, col=1)
                fig.update_yaxes(title_text="Volume", row=current_row, col=1)

            # 3. RSI Subplot
            if show_rsi and 'RSI' in df.columns:
                current_row += 1
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['RSI'], mode='lines',
                    name='RSI (14)', line=dict(color='#E91E63', width=1.5)
                ), row=current_row, col=1)
                fig.add_hline(y=70, line=dict(color='red', width=1, dash='dot'),
                              row=current_row, col=1)
                fig.add_hline(y=30, line=dict(color='green', width=1, dash='dot'),
                              row=current_row, col=1)
                fig.update_yaxes(title_text="RSI", row=current_row, col=1, range=[0, 100])

            # 4. MACD Subplot
            if show_macd and 'MACD' in df.columns:
                current_row += 1
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['MACD'], mode='lines',
                    name='MACD', line=dict(color='#2196F3', width=1.5)
                ), row=current_row, col=1)
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['MACD_Signal'], mode='lines',
                    name='Signal', line=dict(color='#FF9800', width=1.5)
                ), row=current_row, col=1)

                # MACD Histogram
                macd_hist = df['MACD'] - df['MACD_Signal']
                hist_colors = ['#4CAF50' if val >= 0 else '#F44336' for val in macd_hist]
                fig.add_trace(go.Bar(
                    x=df['timestamp'], y=macd_hist, name='Histogram',
                    marker_color=hist_colors
                ), row=current_row, col=1)
                fig.update_yaxes(title_text="MACD", row=current_row, col=1)

            # Update layout
            total_height = 500 if n_rows == 1 else 600 + (n_rows - 1) * 100
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=total_height,
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1)
            )
            fig.update_yaxes(title_text="Price (USD)", tickprefix="$", row=1, col=1)

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering chart: {e}")

        # ═══════════════════════════════════════════════════════════
        #  AI PREDICTION ENGINE
        # ═══════════════════════════════════════════════════════════
        if st.session_state.get('run_prediction', False):
            st.markdown("---")
            st.header("🤖 AI Prediction Engine")

            with st.spinner(f"Loading {selected_model} model for {asset_selection}..."):
                try:
                    scaler = load_scaler(asset_selection)
                    if scaler is None:
                        st.error(f"Scaler not found for {asset_selection}. Run preprocessing first.")
                        st.stop()

                    full_df = load_scaled_data(asset_selection)
                    if full_df is None:
                        st.error(f"Scaled data not found for {asset_selection}.")
                        st.stop()

                    seq_len = BEST_LSTM_CONFIG['seq_len']
                    n_features = full_df.shape[1]
                    target_idx = (list(full_df.columns).index('log_return')
                                  if 'log_return' in full_df.columns
                                  else list(full_df.columns).index('price'))

                    # ── Run prediction with selected model ──
                    pred_scaled = run_model_prediction(
                        asset_selection, selected_model, full_df, scaler
                    )

                    if pred_scaled is not None:
                        # Reconstruct to USD
                        last_seq = full_df.values[-seq_len:].reshape(1, seq_len, n_features)
                        tomorrow_usd = reconstruct_price(
                            np.array([[pred_scaled]]), last_seq, scaler, target_idx
                        )[0]

                        predicted_change = tomorrow_usd - latest_price
                        pct_pred_change = (predicted_change / latest_price) * 100

                        st.subheader("Tomorrow's Forecast")

                        direction = "📈 UP" if predicted_change > 0 else "📉 DOWN"
                        st.info(
                            f"The **{selected_model}** model predicts {asset_selection} "
                            f"will move **{direction}** to **${tomorrow_usd:,.2f}** "
                            f"on the next trading day."
                        )

                        pred_col1, pred_col2 = st.columns(2)
                        with pred_col1:
                            st.metric(
                                label=f"Predicted Close ({selected_model})",
                                value=f"${tomorrow_usd:,.2f}",
                                delta=f"{predicted_change:+,.2f} ({pct_pred_change:+.2f}%)"
                            )
                        with pred_col2:
                            st.metric(
                                label="Current Close",
                                value=f"${latest_price:,.2f}"
                            )

                        # ── Multi-model comparison ──
                        if len(available_models) > 1:
                            st.markdown("---")
                            st.subheader("📊 Multi-Model Comparison")
                            st.markdown("Predictions from all available models for this asset:")

                            comparison_data = []
                            for m_name in available_models:
                                m_pred = run_model_prediction(
                                    asset_selection, m_name, full_df, scaler
                                )
                                if m_pred is not None:
                                    m_usd = reconstruct_price(
                                        np.array([[m_pred]]), last_seq,
                                        scaler, target_idx
                                    )[0]
                                    m_change = ((m_usd - latest_price) / latest_price) * 100
                                    comparison_data.append({
                                        'Model': m_name,
                                        'Predicted Price': f"${m_usd:,.2f}",
                                        'Change': f"{m_change:+.2f}%",
                                        'Direction': "📈" if m_change > 0 else "📉"
                                    })

                            if comparison_data:
                                st.dataframe(
                                    pd.DataFrame(comparison_data),
                                    use_container_width=True,
                                    hide_index=True
                                )

                        st.markdown("---")

                        # ── Walk-forward visualization on recent period ──
                        if selected_model in ('LSTM', 'GRU'):
                            st.subheader("Model Validation: Recent Performance")
                            st.markdown(
                                "Walk-forward predictions vs actual prices over the "
                                "last 150 days."
                            )

                            model_path = os.path.join(
                                MODELS_DIR,
                                f'{get_prefix(asset_selection)}_{selected_model.lower()}_final.keras'
                            )
                            model = load_keras_model(model_path)
                            if model is not None:
                                recent_df = full_df.tail(150 + seq_len)
                                X_recent, y_recent = create_sequences(
                                    recent_df, seq_len=seq_len
                                )

                                y_pred_scaled = model.predict(X_recent, verbose=0)
                                y_recent_usd = reconstruct_price(
                                    y_recent, X_recent, scaler, target_idx
                                )
                                y_pred_usd = reconstruct_price(
                                    y_pred_scaled, X_recent, scaler, target_idx
                                )
                                recent_dates = recent_df.index[seq_len:]

                                fig_pred = go.Figure()
                                fig_pred.add_trace(go.Scatter(
                                    x=recent_dates, y=y_recent_usd,
                                    mode='lines', name='Actual Price',
                                    line=dict(color=color_map[asset_selection], width=2)
                                ))
                                fig_pred.add_trace(go.Scatter(
                                    x=recent_dates, y=y_pred_usd,
                                    mode='lines', name=f'{selected_model} Predicted',
                                    line=dict(color='#00ffcc', width=2, dash='dash')
                                ))

                                fig_pred.update_layout(
                                    template="plotly_dark",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    xaxis=dict(title="Date", showgrid=False),
                                    yaxis=dict(
                                        title="Price (USD)", showgrid=True,
                                        gridcolor="rgba(255,255,255,0.1)",
                                        tickprefix="$"
                                    ),
                                    height=400,
                                    margin=dict(l=0, r=0, t=30, b=0),
                                    hovermode="x unified",
                                    legend=dict(
                                        orientation="h", yanchor="bottom", y=1.02,
                                        xanchor="right", x=1
                                    )
                                )
                                st.plotly_chart(fig_pred, use_container_width=True)

                    else:
                        st.error(
                            f"Could not generate prediction with {selected_model}. "
                            f"Model file may be missing or incompatible."
                        )

                except Exception as e:
                    st.error(f"An error occurred during AI Inference: {e}")
                    logger.exception("Dashboard prediction error")
    else:
        st.info("No historical data found. Please run data collection and preprocessing first.")

with tab2:
    st.header("⚙️ Model Performance Metrics")
    perf_df = load_performance_metrics()

    if perf_df is not None:
        try:
            asset_metrics = perf_df[perf_df['Asset'] == asset_selection]

            if not asset_metrics.empty:
                # Summary metrics for the best model
                best_row = asset_metrics.loc[asset_metrics['RMSE'].idxmin()]

                st.subheader(f"Best Model: {best_row['Model']}")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("RMSE", f"${best_row['RMSE']:,.2f}")
                with m2:
                    st.metric("MAE", f"${best_row['MAE']:,.2f}")
                with m3:
                    st.metric("MAPE", f"{best_row['MAPE']:.2f}%")
                with m4:
                    r2_val = best_row['R2']
                    st.metric("R² Score", f"{r2_val:.4f}")

                st.markdown("---")
                st.subheader("All Models Comparison (RMSE)")
                st.markdown("Lower RMSE is better. Models ranked by prediction accuracy.")

                # Prepare chart
                chart_df = asset_metrics.sort_values('RMSE', ascending=False)

                # Assign colors
                model_colors = {
                    'LSTM (Optimized)': '#2196F3',
                    'GRU': '#00BCD4',
                    'LightGBM': '#8BC34A',
                    'CatBoost': '#FF5722',
                    'Ensemble (0.3A+0.7L)': '#E91E63',
                    'Naive (t=t-1)': '#9E9E9E',
                    'Linear Regression': '#4CAF50',
                    'Random Forest': '#795548',
                    'ARIMA(1,1,0)': '#FF9800',
                }
                colors = [model_colors.get(m, '#607D8B') for m in chart_df['Model']]

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
                    height=max(300, len(chart_df) * 50),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                fig_bar.update_traces(texttemplate='$%{x:,.0f}', textposition='outside')
                st.plotly_chart(fig_bar, use_container_width=True)

                # Full metrics table
                st.markdown("---")
                st.subheader("Full Metrics Table")
                display_df = asset_metrics[['Model', 'RMSE', 'MAE', 'MAPE', 'R2', 'Dir_Acc']].copy()
                display_df = display_df.sort_values('RMSE')
                display_df.columns = ['Model', 'RMSE ($)', 'MAE ($)', 'MAPE (%)', 'R²', 'Dir. Accuracy (%)']
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                st.info(
                    "💡 **Note on Naive Baseline:** In financial time series, the 'Naive' forecast "
                    "(tomorrow = today) often produces low raw error because daily moves are small. "
                    "The key metric is whether intelligent models learn useful patterns beyond this baseline."
                )
            else:
                st.warning("No metrics found for this asset. Run backtesting first.")
        except Exception as e:
            st.error(f"Error rendering performance metrics: {e}")
    else:
        st.warning("Performance metrics not found. Run `python src/evaluation/backtesting.py` first.")

with tab3:
    st.header("📋 Model Inventory")

    for asset_name in ["Bitcoin", "Gold", "Silver"]:
        prefix = get_prefix(asset_name)
        status = MODEL_STATUS.get(asset_name, {})
        models = get_available_models(asset_name)

        with st.expander(f"**{asset_name}** — {len(models)} models available", expanded=(asset_name == asset_selection)):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Primary Model:** `{status.get('primary_model', 'N/A')}`")
                st.markdown(f"**Status:** `{status.get('status', 'unknown')}`")
            with col_b:
                st.markdown("**Available Models:**")
                for m_name, m_path in models.items():
                    file_size = os.path.getsize(m_path) / 1024
                    st.markdown(f"- ✅ `{m_name}` ({file_size:.0f} KB)")

            # Check for scaler
            scaler_path = os.path.join(MODELS_DIR, f'{prefix}_scaler.pkl')
            if os.path.exists(scaler_path):
                st.markdown("- ✅ Scaler available")
            else:
                st.markdown("- ❌ **Scaler missing** — predictions will fail")
