"""
Streamlit Dashboard — Bitcoin / Gold / Silver next-day forecasting.

Tabs:
  1. Forecast & Indicators   price chart with BB / RSI / MACD overlays, next-day prediction
                             with uncertainty band, multi-model comparison, disclaimer
  2. Model Performance       walk-forward validation table, held-out test table, figures
  3. Methodology & Models    pipeline, features, served-model details, model inventory
"""
import os
import sys
import json
import logging
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR, SEQ_LEN, PREDICTION_HORIZON_DAYS,
                    TRAIN_END, VAL_END, ASSETS, get_prefix, load_model_status)
from src.inference.prediction import (predict_next_day, predict_for_date, predict_all_models, available_models,
                                      prediction_history, DISCLAIMER)
from src.data.sync_live_data import update_live_data

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Crypto & Metals AI Predictor", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  .stApp { background-color: #0E1117; }
  h1, h2, h3 { font-family: 'Inter', sans-serif; color: #FFFFFF; font-weight: 700; letter-spacing: -0.5px; }
  div[data-testid="stMetricValue"] { font-size: 2.0rem; }
</style>""", unsafe_allow_html=True)

COLOR = {"Bitcoin": "#F7931A", "Gold": "#FFD700", "Silver": "#C0C0C0"}


# ─────────────────────────────────────────────── cached loaders
@st.cache_data(ttl=600)
def load_features(asset: str) -> Optional[pd.DataFrame]:
    p = get_prefix(asset)
    live = os.path.join(PROCESSED_DATA_DIR, f'{p}_live_features.csv')
    path = live if os.path.exists(live) else os.path.join(PROCESSED_DATA_DIR, f'{p}_features.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp')


@st.cache_data(ttl=600)
def load_csv(name: str) -> Optional[pd.DataFrame]:
    path = os.path.join(RESULTS_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data(ttl=600)
def load_test_predictions(asset: str) -> Optional[pd.DataFrame]:
    path = os.path.join(RESULTS_DIR, 'predictions', f'{get_prefix(asset)}_test_predictions.csv')
    return pd.read_csv(path, parse_dates=['date', 'target_date']) if os.path.exists(path) else None


def fmt_pct(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1f}%"


# ─────────────────────────────────────────────── sidebar
st.sidebar.title("AI Prediction Engine")
# Optional deep-link parameters for demos: ?asset=Gold&run=1
_qp = st.query_params
_default_asset = ASSETS.index(_qp.get('asset')) if _qp.get('asset') in ASSETS else 0
asset = st.sidebar.selectbox("Target Asset", ASSETS, index=_default_asset)
if _qp.get('run') == '1':
    st.session_state['run_prediction'] = True
status_all = load_model_status()
status = status_all.get(asset, {})
served = status.get('primary_model', 'LightGBM')
models = available_models(asset)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Sync Live Market Data", use_container_width=True):
    with st.spinner("Fetching latest data from Yahoo Finance..."):
        ok = update_live_data(asset, refresh_external=True)
    if ok:
        st.cache_data.clear(); st.sidebar.success("Live data synced."); st.rerun()
    else:
        st.sidebar.warning("Yahoo Finance did not return data (rate limit). Using the stored dataset.")

st.sidebar.markdown("---")
st.sidebar.subheader("Technical Indicators")
show_bb = st.sidebar.checkbox("Bollinger Bands")
show_rsi = st.sidebar.checkbox("RSI (14)")
show_macd = st.sidebar.checkbox("MACD (12, 26, 9)")
show_volume = st.sidebar.checkbox("Volume", value=(asset == 'Bitcoin'))

st.sidebar.markdown("---")
st.sidebar.subheader("AI Model")
default_idx = models.index(served) if served in models else 0
selected_model = st.sidebar.selectbox("Select Model", models or ['(no trained models)'], index=default_idx)
if st.sidebar.button("🚀 Run AI Prediction", use_container_width=True):
    st.session_state['run_prediction'] = True
st.session_state.setdefault('run_prediction', False)

st.sidebar.markdown("---")
st.sidebar.info(f"**Served model:** {served}\n\n**Horizon:** {PREDICTION_HORIZON_DAYS} trading day\n\n"
                f"**Lookback:** {SEQ_LEN} days\n\n**Selected by:** {status.get('selection_rule', 'n/a')}")

# ─────────────────────────────────────────────── main
st.title(f"📈 {asset} — Next-Day Forecast Dashboard")
tab1, tab_demo, tab_hist, tab2, tab3 = st.tabs(["📊 Forecast & Indicators", "🎯 Predict a Day (unseen test)",
                                                  "🗂️ Prediction History", "⚙️ Model Performance", "📋 Methodology & Models"])
df = load_features(asset)

with tab1:
    if df is None or len(df) < 2:
        st.info("No processed data found. Run `python src/data/preprocessing.py` first.")
    else:
        latest_date = df['timestamp'].iloc[-1].strftime("%Y-%m-%d")
        latest_price, prev_price = df['price'].iloc[-1], df['price'].iloc[-2]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Latest Close ({latest_date})", f"${latest_price:,.2f}",
                  f"{latest_price - prev_price:+,.2f} ({(latest_price / prev_price - 1) * 100:+.2f}%)")
        c2.metric("30-Day High", f"${df['price'].tail(30).max():,.2f}")
        c3.metric("30-Day Low", f"${df['price'].tail(30).min():,.2f}")
        c4.metric("30-Day Volatility (daily σ)", f"{df['log_return'].tail(30).std() * 100:.2f}%")
        st.markdown("---")

        # ── price chart with indicators (unchanged from the original dashboard)
        st.subheader(f"Historical {asset} Price & Technical Analysis")
        n_rows, heights = 1, [0.55]
        for flag, h in ((show_volume, 0.15), (show_rsi, 0.15), (show_macd, 0.2)):
            if flag:
                n_rows += 1; heights.append(h)
        if n_rows == 1:
            heights = [1.0]
        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=heights)
        rgb = tuple(int(COLOR[asset].lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'], mode='lines', name=f'{asset} close',
                                 line=dict(color=COLOR[asset], width=2), fill='tozeroy',
                                 fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.1)"), row=1, col=1)
        t_end, v_end = pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END)
        for x0, x1, c, n in ((df['timestamp'].min(), t_end, '#4CAF50', 'train'), (t_end, v_end, '#FFC107', 'validation'),
                             (v_end, df['timestamp'].max(), '#F44336', 'test')):
            fig.add_vrect(x0=x0, x1=x1, fillcolor=c, opacity=0.06, line_width=0, annotation_text=n, annotation_position='top left', row=1, col=1)
        if show_bb:
            for col, nm, dash in (('BB_Upper', 'BB upper', 'dash'), ('BB_Lower', 'BB lower', 'dash'), ('BB_Mid', 'BB mid (SMA20)', 'dot')):
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], mode='lines', name=nm, line=dict(color='gray', width=1, dash=dash)), row=1, col=1)
        row = 1
        if show_volume and 'volume' in df.columns:
            row += 1
            colors = np.where(df['price'].diff().fillna(0) >= 0, '#4CAF50', '#F44336')
            fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color=colors, opacity=0.6), row=row, col=1)
        if show_rsi:
            row += 1
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], mode='lines', name='RSI (14)', line=dict(color='#E91E63', width=1.5)), row=row, col=1)
            fig.add_hline(y=70, line=dict(color='red', width=1, dash='dot'), row=row, col=1)
            fig.add_hline(y=30, line=dict(color='green', width=1, dash='dot'), row=row, col=1)
        if show_macd:
            row += 1
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MACD'], mode='lines', name='MACD', line=dict(color='#2196F3', width=1.5)), row=row, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MACD_Signal'], mode='lines', name='Signal', line=dict(color='#FF9800', width=1.5)), row=row, col=1)
            hist = df['MACD'] - df['MACD_Signal']
            fig.add_trace(go.Bar(x=df['timestamp'], y=hist, name='Histogram', marker_color=np.where(hist >= 0, '#4CAF50', '#F44336')), row=row, col=1)
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          height=500 if n_rows == 1 else 600 + (n_rows - 1) * 90, margin=dict(l=0, r=0, t=30, b=0),
                          hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_yaxes(title_text="Price (USD)", tickprefix="$", row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # ── prediction
        if st.session_state['run_prediction'] and models:
            st.markdown("---")
            st.header("🤖 Next-Day Prediction")
            try:
                with st.spinner(f"Running {selected_model}..."):
                    r = predict_next_day(asset, selected_model)
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Current close", f"${r['current_price']:,.2f}", help=f"as of {r['as_of_date']}")
                p2.metric(f"Predicted close (T+{r['horizon_days']})", f"${r['predicted_price']:,.2f}",
                          f"{r['predicted_price'] - r['current_price']:+,.2f} ({r['predicted_return_pct']:+.2f}%)")
                p3.metric("Direction", "📈 UP" if r['direction'] == 'UP' else "📉 DOWN")
                p4.metric("Model", r['model_used'], help="served model" if r['is_served_model'] else "alternative model")
                if r['uncertainty_band']:
                    lo, hi = r['uncertainty_band']
                    st.caption(f"±1 RMSE band from the held-out test period: **\\${lo:,.2f} – \\${hi:,.2f}**. "
                               f"The band is wider than the predicted move, which is the honest picture of next-day uncertainty.")
                tm = r.get('test_metrics') or {}
                if tm:
                    st.markdown(
                        f"**Held-out test performance of {r['model_used']}** ({r['test_period'][0]} → {r['test_period'][1]}, {tm.get('n_test')} days): "
                        f"MAE \\${tm.get('MAE_usd', float('nan')):,.2f} · RMSE \\${tm.get('RMSE_usd', float('nan')):,.2f} · MAPE {tm.get('MAPE_usd', float('nan')):.2f}% · "
                        f"R²(return) {tm.get('R2_ret', float('nan')):+.3f} · directional accuracy **{fmt_pct(tm.get('DirAcc_pct'))}** "
                        f"(p = {tm.get('DirAcc_pvalue', float('nan')):.2f}) · Diebold–Mariano vs random walk p = {tm.get('DM_pvalue', float('nan')):.2f}. "
                        f"Random-walk RMSE on the same days: \\${(r.get('test_naive') or {}).get('RMSE_usd', float('nan')):,.2f}.")
                st.warning(DISCLAIMER)

                # forecast horizon visual: last 60 days + T+1 point with band
                tail = df.tail(60)
                next_date = tail['timestamp'].iloc[-1] + pd.tseries.offsets.BDay(1) if asset != 'Bitcoin' else tail['timestamp'].iloc[-1] + pd.Timedelta(days=1)
                fh = go.Figure()
                fh.add_trace(go.Scatter(x=tail['timestamp'], y=tail['price'], mode='lines', name='close', line=dict(color=COLOR[asset], width=2)))
                if r['uncertainty_band']:
                    fh.add_trace(go.Scatter(x=[next_date, next_date], y=r['uncertainty_band'], mode='lines', name='±1 RMSE band',
                                            line=dict(color='rgba(255,255,255,0.4)', width=6)))
                fh.add_trace(go.Scatter(x=[tail['timestamp'].iloc[-1], next_date], y=[r['current_price'], r['predicted_price']], mode='lines+markers',
                                        name=f'{selected_model} forecast', line=dict(color='#00ffcc', width=2, dash='dash'), marker=dict(size=9)))
                fh.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=380,
                                 margin=dict(l=0, r=0, t=30, b=0), yaxis=dict(tickprefix="$"), title="Forecast horizon: last 60 days + next trading day")
                st.plotly_chart(fh, use_container_width=True)

                # multi-model comparison
                if len(models) > 1:
                    st.subheader("📊 All trained models — same input window")
                    rows = predict_all_models(asset)
                    if rows:
                        t = pd.DataFrame(rows)
                        t['Predicted Price'] = t['predicted_price'].map(lambda v: f"${v:,.2f}")
                        t['Change'] = t['predicted_return_pct'].map(lambda v: f"{v:+.2f}%")
                        t['Dir'] = t['direction'].map({'UP': '📈', 'DOWN': '📉'})
                        t['Served'] = t['model'].map(lambda m: '✅' if m == served else '')
                        st.dataframe(t[['model', 'Predicted Price', 'Change', 'Dir', 'Served']], use_container_width=True, hide_index=True)
                        st.caption("Disagreement between models is expected: daily returns are mostly noise. "
                                   "Only the served model has been selected on validation data; the others are comparison rows.")

                # recent performance of the selected model on the test period
                tp = load_test_predictions(asset)
                if tp is not None and f'pred_return_{selected_model}' in tp.columns:
                    st.subheader(f"Model validation: {selected_model} on the untouched test period")
                    fr = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.5, 0.5],
                                       subplot_titles=("Next-day log return: actual vs predicted (%)", "Price: actual vs predicted"))
                    fr.add_trace(go.Scatter(x=tp['target_date'], y=tp['actual_return'] * 100, name='actual return', line=dict(color='white', width=1)), row=1, col=1)
                    fr.add_trace(go.Scatter(x=tp['target_date'], y=tp[f'pred_return_{selected_model}'] * 100, name=f'{selected_model} return', line=dict(color='#00ffcc', width=1.5)), row=1, col=1)
                    fr.add_trace(go.Scatter(x=tp['target_date'], y=tp['actual_close'], name='actual close', line=dict(color=COLOR[asset], width=2)), row=2, col=1)
                    fr.add_trace(go.Scatter(x=tp['target_date'], y=tp[f'pred_close_{selected_model}'], name=f'{selected_model} close', line=dict(color='#00ffcc', width=1.5, dash='dash')), row=2, col=1)
                    fr.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=620,
                                     margin=dict(l=0, r=0, t=40, b=0), hovermode="x unified")
                    st.plotly_chart(fr, use_container_width=True)
                    sc = go.Figure()
                    sc.add_trace(go.Scatter(x=tp['actual_return'] * 100, y=tp[f'pred_return_{selected_model}'] * 100, mode='markers',
                                            marker=dict(color='#00ffcc', size=7, opacity=0.7), name='days'))
                    lim = float(np.abs(tp['actual_return'] * 100).max()) * 1.05
                    sc.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode='lines', line=dict(color='red', dash='dash'), name='perfect prediction'))
                    sc.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=380,
                                     xaxis_title="actual return (%)", yaxis_title="predicted return (%)", margin=dict(l=0, r=0, t=30, b=0),
                                     title="True skill view: predicted vs actual returns (a price chart hides this)")
                    st.plotly_chart(sc, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                logger.exception("Dashboard prediction error")

# ─────────────────────────────────────────────── demo: predict any day of the unseen test period
with tab_demo:
    st.header("🎯 Predict a Day — unseen test period")
    st.markdown(
        f"Pick a day. The model sees data **only up to that day**, predicts the next trading day's close, "
        f"and then the actual close is revealed. Days after **{VAL_END}** were never used for training, "
        f"validation or model selection (frozen chronological split).")
    hist = prediction_history(asset)
    if hist is None or df is None:
        st.warning("Run `python src/evaluation/backtesting.py` first to generate the unseen-test predictions.")
    else:
        test_days = list(hist['date'].dt.date)
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            pick = st.selectbox("Stand on this day (data available up to and including it)", test_days, index=len(test_days) - 1,
                                format_func=lambda d: d.strftime('%Y-%m-%d (%a)'))
        with c2:
            demo_model = st.selectbox("Model", models, index=models.index(served) if served in models else 0, key='demo_model')
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            go_demo = st.button("🎯 Generate Prediction", use_container_width=True)
        if go_demo or st.session_state.get('demo_done'):
            st.session_state['demo_done'] = True
            try:
                r = predict_for_date(asset, str(pick), demo_model)
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric(f"Close on {r['as_of_date']}", f"${r['current_price']:,.2f}")
                m2.metric(f"Predicted close for {r['actual_date'] or 'next day'}", f"${r['predicted_price']:,.2f}",
                          f"{r['predicted_return_pct']:+.2f}%")
                if r['actual_price'] is not None:
                    m3.metric("Actual close", f"${r['actual_price']:,.2f}", f"{r['actual_return_pct']:+.2f}%")
                    m4.metric("Prediction error", f"{r['error_pct']:+.2f}%", f"{r['error_usd']:+,.2f} USD", delta_color="off")
                    hit = r['direction_hit']
                    m5.metric("Direction", f"{r['direction']} → {'✅ hit' if hit else ('❌ miss' if hit is not None else '—')}")
                else:
                    m3.metric("Actual close", "not yet known")
                st.caption(f"Model: **{r['model_used']}** · horizon: {r['horizon_days']} trading day · "
                           f"{'inside the unseen test period' if r['in_unseen_test_period'] else 'inside the training/validation period'} · "
                           "the prediction uses only data up to the chosen day.")

                # chart: unseen period, actual vs predicted, highlighted day
                col_pred = f'pred_close_{demo_model}'
                if col_pred in hist.columns:
                    fd = go.Figure()
                    fd.add_trace(go.Scatter(x=hist['target_date'], y=hist['actual_close'], mode='lines', name='actual close',
                                            line=dict(color=COLOR[asset], width=2)))
                    fd.add_trace(go.Scatter(x=hist['target_date'], y=hist[col_pred], mode='lines', name=f'{demo_model} predicted close',
                                            line=dict(color='#00ffcc', width=1.5, dash='dash')))
                    if r['actual_price'] is not None:
                        fd.add_trace(go.Scatter(x=[pd.Timestamp(r['actual_date'])], y=[r['predicted_price']], mode='markers', name='this prediction',
                                                marker=dict(color='#00ffcc', size=14, symbol='diamond', line=dict(color='white', width=1))))
                        fd.add_trace(go.Scatter(x=[pd.Timestamp(r['actual_date'])], y=[r['actual_price']], mode='markers', name='actual',
                                                marker=dict(color=COLOR[asset], size=14, symbol='circle', line=dict(color='white', width=1))))
                    fd.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=420,
                                     margin=dict(l=0, r=0, t=40, b=0), hovermode="x unified", yaxis=dict(tickprefix="$"),
                                     title=f"Unseen test period {hist['target_date'].min().date()} → {hist['target_date'].max().date()}: actual vs predicted ({demo_model})")
                    st.plotly_chart(fd, use_container_width=True)

                # all models on this day
                rows = predict_all_models(asset, str(pick))
                if rows:
                    t = pd.DataFrame(rows)
                    t['Predicted'] = t['predicted_price'].map(lambda v: f"${v:,.2f}")
                    t['Actual'] = t['actual_price'].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
                    t['Error'] = t['error_pct'].map(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
                    t['Served'] = t['model'].map(lambda m: '✅' if m == served else '')
                    st.dataframe(t[['model', 'Predicted', 'Actual', 'Error', 'Served']], use_container_width=True, hide_index=True)
                st.warning(DISCLAIMER)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                logger.exception("Demo prediction error")

# ─────────────────────────────────────────────── prediction history (out-of-sample log)
with tab_hist:
    st.header("🗂️ Prediction History — every unseen test day")
    hist = prediction_history(asset)
    if hist is None:
        st.warning("Run `python src/evaluation/backtesting.py` first.")
    else:
        hm = st.selectbox("Model", [m for m in models if f'pred_close_{m}' in hist.columns] + ['Naive'],
                          index=0 if served not in models else models.index(served), key='hist_model')
        h = pd.DataFrame({'Predicted on': hist['date'].dt.date, 'For': hist['target_date'].dt.date,
                          'Predicted ($)': hist[f'pred_close_{hm}'].round(2), 'Actual ($)': hist['actual_close'].round(2),
                          'Error (%)': hist[f'error_pct_{hm}'].round(2), 'Direction hit': hist[f'direction_hit_{hm}'].map({1: '✅', 0: '❌'})})
        e = hist[f'error_pct_{hm}'].abs()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Unseen days", len(h))
        k2.metric("Mean absolute error", f"{e.mean():.2f}%")
        k3.metric("Median absolute error", f"{e.median():.2f}%")
        k4.metric("Direction hit rate", f"{hist[f'direction_hit_{hm}'].mean() * 100:.1f}%")
        st.dataframe(h.sort_values('For', ascending=False), use_container_width=True, hide_index=True, height=480)
        st.download_button("Download history (CSV)", h.to_csv(index=False).encode(), file_name=f"{asset.lower()}_prediction_history.csv")
        st.caption("Every row is an out-of-sample prediction: the model was trained on data up to "
                   f"{VAL_END} and each prediction used only data up to the 'Predicted on' date.")

with tab2:
    st.header("⚙️ Model Performance")
    cv, te = load_csv('cv_results.csv'), load_csv('final_test_results.csv')
    if cv is None or te is None:
        st.warning("Run `python src/evaluation/backtesting.py` to generate results.")
    else:
        s = status
        st.markdown(f"**Served model: `{served}`** — selected by {s.get('selection_rule', 'walk-forward validation')}. "
                    f"Test period {s.get('test_period', ['?', '?'])[0]} → {s.get('test_period', ['?', '?'])[1]}.")
        st.subheader("Walk-forward validation (train + validation period, 4 expanding folds)")
        c = cv[cv['asset'] == asset].sort_values('RMSE_ret_mean')
        show = pd.DataFrame({'Model': c['model'],
                             'RMSE (return)': c['RMSE_ret_mean'].map(lambda v: f"{v:.5f}") + ' ± ' + c['RMSE_ret_std'].map(lambda v: f"{v:.5f}"),
                             'MAE (return)': c['MAE_ret_mean'].round(5), 'R² (return)': c['R2_ret_mean'].round(3),
                             'Dir. Acc %': c['DirAcc_pct_mean'].round(1), 'RMSE ($)': c['RMSE_usd_mean'].round(2)})
        st.dataframe(show, use_container_width=True, hide_index=True)

        st.subheader("Untouched test set (evaluated once)")
        t = te[te['asset'] == asset].sort_values('RMSE_ret')
        show = pd.DataFrame({'Model': t['model'], 'MAE ($)': t['MAE_usd'].round(2), 'RMSE ($)': t['RMSE_usd'].round(2),
                             'MAPE %': t['MAPE_usd'].round(2), 'RMSE (return)': t['RMSE_ret'].round(5), 'R² (return)': t['R2_ret'].round(3),
                             'Dir. Acc %': t['DirAcc_pct'].round(1), 'DA p-value': t['DirAcc_pvalue'].round(3),
                             'DM p vs naive': t['DM_pvalue'].round(3), 'Strategy %': t['strategy_return_pct'].round(1),
                             'Buy&Hold %': t['buy_hold_return_pct'].round(1)})
        st.dataframe(show, use_container_width=True, hide_index=True)
        naive = t[t['model'] == 'Naive'].iloc[0]; best = t[t['model'] == served].iloc[0]
        rel = (best['RMSE_ret'] / naive['RMSE_ret'] - 1) * 100
        if best['DM_pvalue'] < 0.05 and rel < 0:
            verdict = f"it beats the naive baseline on magnitude ({rel:+.2f}%, Diebold–Mariano p = {best['DM_pvalue']:.2f}, significant at 5%)"
        elif abs(rel) < 0.05:
            verdict = f"it is indistinguishable from the naive baseline on magnitude (Diebold–Mariano p = {best['DM_pvalue']:.2f})"
        else:
            verdict = (f"it is {abs(rel):.2f}% {'below' if rel < 0 else 'above'} the naive baseline on magnitude — "
                       f"not a significant difference (Diebold–Mariano p = {best['DM_pvalue']:.2f})")
        da_sig = 'significant' if best['DirAcc_pvalue'] < 0.05 else 'not significant'
        st.info(f"**Honest reading:** on the test set the served model's RMSE(return) is {best['RMSE_ret']:.5f} vs {naive['RMSE_ret']:.5f} for the "
                f"random-walk forecast — {verdict}. "
                f"Directional accuracy is {best['DirAcc_pct']:.1f}% on {int(best['DirAcc_n'])} non-flat days (binomial p = {best['DirAcc_pvalue']:.2f}, {da_sig}). "
                "R² on price levels is intentionally not shown: a random walk scores ≈0.95 there.")
        reg = load_csv('regime_analysis.csv')
        if reg is not None:
            st.subheader("Performance across market regimes (unseen test period)")
            rg = reg[reg['asset'] == asset].copy()
            rg = pd.DataFrame({'Regime': rg['regime'], 'Days': rg['n_days'], f'MAE % ({served})': rg['mae_pct_served'].round(2),
                               'MAE % (naive)': rg['mae_pct_naive'].round(2), f'Direction hit % ({served})': rg['dir_hit_served_pct'].round(1),
                               'RMSE ret (served)': rg['rmse_ret_served'].round(5), 'RMSE ret (naive)': rg['rmse_ret_naive'].round(5)})
            st.dataframe(rg, use_container_width=True, hide_index=True)
            st.caption("Regimes are defined on the day the prediction is made (30-day volatility tercile, sign of the 20-day return), "
                       "so they never use future information.")
        with st.expander("Design experiments (validation only): data size, feature ablation, horizon, volatility"):
            for fname, title in (('E1_data_size.csv', 'E1 — training-data size (fixed validation window)'),
                                 ('E2_feature_ablation.csv', 'E2 — feature-group ablation (walk-forward CV)'),
                                 ('E3_horizon.csv', 'E3 — next-day vs 5-day return target'),
                                 ('E4_volatility.csv', 'E4 — 22-day realised-volatility target')):
                ex = load_csv(os.path.join('experiments', fname))
                if ex is not None:
                    st.markdown(f"**{title}**")
                    st.dataframe(ex[ex['asset'] == asset].round(4), use_container_width=True, hide_index=True)
        for name in ('model_comparison_test.png', f'{get_prefix(asset)}_actual_vs_predicted.png', f'{get_prefix(asset)}_feature_importance.png',
                     f'{get_prefix(asset)}_loss_curves.png', f'{get_prefix(asset)}_strategy.png', 'overfitting_gap.png'):
            p = os.path.join(FIGURES_DIR, name)
            if os.path.exists(p):
                st.image(p, use_container_width=True)

with tab3:
    st.header("📋 Methodology & Model Inventory")
    st.markdown(f"""
**Task.** Predict the next trading day's log return, `y_t = ln(P_(t+1) / P_t)`, and convert it to a price with `P̂_(t+1) = P_t · exp(ŷ_t)`.
Predicting the return (not the price) makes the target stationary and makes the random-walk baseline explicit.

**Data.** Yahoo Finance daily data from {status.get('data_start', '2018')} ({status.get('n_train', '?')} training days). **Split (frozen, chronological, no shuffling).** train ≤ {TRAIN_END} · validation ≤ {VAL_END} · test = remainder.
The scaler is fitted on train only; hyper-parameters and the served model are chosen by 4-fold expanding-window
walk-forward validation inside train+val; the test set is evaluated once by `src/evaluation/backtesting.py`.

**Features ({len(status.get('features', []))}).** Stationary, backward-looking only: today's and lagged log returns, rolling volatility,
RSI, ADX, ROC, MACD/price, price/EMA ratios, Bollinger %B and width, ATR/price, high–low range,
macro daily returns (S&P 500, VIX{', DXY, oil, 10-year yield' if asset != 'Bitcoin' else ', Fear & Greed, day-of-week, volume change'}){', gold return' if asset == 'Silver' else ''}.
Raw price levels are shown on charts but never fed to a model.
{'**Close-time rule.** Gold/Silver close at the 13:30 ET COMEX settlement, before the US equity/rates/FX closes, so the macro returns and the High/Low-based features (hl_range, ATR/price, ADX) are the *previous* session' + chr(39) + 's values — only information known when the price is fixed is used.' if asset != 'Bitcoin' else '**Close-time rule.** BTC-USD closes at 00:00 UTC, after every US market close, so same-day macro values are known at the close.'}

**Models compared.** Naive zero-return (random walk), historical mean, ARIMA, Ridge, Random Forest, LightGBM, CatBoost, GRU, LSTM, and a stacked ensemble.
Models needing a stopping rule (trees / epochs) use validation for that in phase 1 and are refit on train+val in phase 2.
""")
    st.markdown("**Served model details**")
    st.json({k: status.get(k) for k in ('primary_model', 'selection_rule', 'cv_rmse_ret', 'cv_rmse_ret_naive', 'cv_dir_acc', 'params', 'fit_info')})
    st.markdown("**Model inventory**")
    for a in ASSETS:
        with st.expander(f"{a} — {len(available_models(a))} trained models", expanded=(a == asset)):
            st.markdown(f"Served: `{status_all.get(a, {}).get('primary_model', 'n/a')}` · available: {', '.join(available_models(a)) or 'none'}")
    st.warning(DISCLAIMER)
