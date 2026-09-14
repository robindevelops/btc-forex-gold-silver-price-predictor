"""
Streamlit dashboard — next-day price forecasting for Bitcoin, Gold and Silver.

Every number shown is read from the project's own artefacts (data/processed, data/models, results/) or
produced live by src/inference/prediction.py. Nothing is hard-coded.
The user never chooses a model: "Run prediction" runs every trained model and serves their combination.

Layout
  Overview strip            what is predicted · the served forecast · its reliability on the unseen test period
  1 Forecast                latest market data · Run prediction → one combined next-day forecast (+ each model) · charts
  2 Predict a Day           stand on any unseen test day, predict, reveal the actual close and the error
  3 Test-Set History        every unseen test day: predicted vs actual, error, direction hit (CSV download)
  4 Model Performance       test table · honest reading · comparison chart · walk-forward table · predicted vs actual
  5 Methodology             target, data, split, features, models, how the combination works
"""
import os
import sys
import time
import logging
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR, PREDICTION_HORIZON_DAYS,
                    TRAIN_END, VAL_END, ASSETS, ASSET_CONFIG, get_prefix, load_model_status)
from src.inference.prediction import (predict_next_day, predict_for_date, prediction_history, combined_members, live_data_is_current,
                                      COMBINED, DISCLAIMER)
from src.data.market_calendar import expected_last_complete_bar
from src.data.sync_live_data import update_live_data

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Next-Day Price Forecast — Bitcoin, Gold, Silver", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  .stApp { background-color: #0E1117; }
  h1, h2, h3 { font-family: 'Inter', sans-serif; color: #FFFFFF; font-weight: 700; letter-spacing: -0.3px; }
  h1 { font-size: 1.9rem !important; margin-bottom: 0.2rem !important; }
  h2 { font-size: 1.35rem !important; margin-top: 1.2rem !important; }
  h3 { font-size: 1.05rem !important; }
  div[data-testid="stMetricValue"] { font-size: 1.7rem; }
  div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #A0A4AB; }
  .overview { background: #161B24; border: 1px solid #262C38; border-radius: 8px; padding: 0.8rem 1.1rem; margin-bottom: 0.6rem;
              font-size: 0.93rem; line-height: 1.6; color: #D6D9DE; }
  .overview b { color: #FFFFFF; }
  .final { background: linear-gradient(135deg, #12202A 0%, #161B24 100%); border: 1px solid #1F6F66; border-radius: 10px;
           padding: 1rem 1.3rem; margin: 0.4rem 0 0.8rem 0; }
  .final .lbl { color: #A0A4AB; font-size: 0.85rem; }
  .final .big { font-size: 2.1rem; font-weight: 700; color: #FFFFFF; font-family: 'Inter', sans-serif; }
  .final .up { color: #2ECC71; } .final .down { color: #FF5252; }
  .footer { color: #8A8F98; font-size: 0.8rem; margin-top: 2rem; border-top: 1px solid #262C38; padding-top: 0.6rem; }
</style>""", unsafe_allow_html=True)

COLOR = {"Bitcoin": "#F7931A", "Gold": "#FFD700", "Silver": "#C0C0C0"}
ACCENT = "#00E0C6"          # predicted / model colour
NAIVE = "#9AA0A6"           # random-walk colour
BAND = "rgba(255,255,255,0.35)"


# ─────────────────────────────────────────────── data loaders (all from project artefacts)
@st.cache_data(ttl=300)
def load_features(asset: str) -> Optional[pd.DataFrame]:
    p = get_prefix(asset)
    live = os.path.join(PROCESSED_DATA_DIR, f'{p}_live_features.csv')
    path = live if os.path.exists(live) else os.path.join(PROCESSED_DATA_DIR, f'{p}_features.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp')
    df.attrs['source'] = 'live download' if path == live else 'frozen dataset'
    return df


@st.cache_data(ttl=300)
def load_csv(name: str) -> Optional[pd.DataFrame]:
    path = os.path.join(RESULTS_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data(ttl=300)
def load_test_predictions(asset: str) -> Optional[pd.DataFrame]:
    return prediction_history(asset)


def test_row(te: Optional[pd.DataFrame], asset: str, model: str) -> Optional[pd.Series]:
    """One model's single test-set evaluation (results/final_test_results.csv)."""
    if te is None:
        return None
    r = te[(te['asset'] == asset) & (te['model'] == model)]
    return r.iloc[0] if len(r) else None


def layout(fig, height, title=None, legend=True):
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=0, r=0, t=48 if title else 24, b=0), showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=11)))
    if fig.layout.hovermode is None:
        fig.update_layout(hovermode="x unified")
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15)))
    return fig


def pct(x, d=1):
    return "—" if x is None or pd.isna(x) else f"{x:.{d}f}%"


def usd(x):
    return "—" if x is None or pd.isna(x) else f"${x:,.2f}"


def usd_md(x):
    """Dollar amount for st.markdown / st.caption, where a bare '$' would start LaTeX."""
    return usd(x).replace('$', '\\$')


# ─────────────────────────────────────────────── sidebar: asset + the one button
st.sidebar.title("Next-Day Forecast")
_qp = st.query_params                                    # deep links for demos: ?asset=Gold&run=1
_default_asset = ASSETS.index(_qp.get('asset')) if _qp.get('asset') in ASSETS else 0
asset = st.sidebar.selectbox("Asset", ASSETS, index=_default_asset)
status_all = load_model_status()
status = status_all.get(asset, {})
members = combined_members(asset)
cv_best = status.get('primary_model')
if _qp.get('run') == '1':
    st.session_state['run_prediction'] = True
if st.sidebar.button("Run prediction", type="primary", width='stretch',
                     help="Downloads the latest complete bar, runs every trained model and combines them into one forecast."):
    st.session_state['run_prediction'] = True
    st.session_state.pop('run_result', None)
st.session_state.setdefault('run_prediction', False)
st.sidebar.caption(f"Runs **all {len(members)} trained models** ({', '.join(members) or 'none'}) and serves their "
                   f"equal-weight combination. No model to choose.")

st.sidebar.markdown("---")
overlay_options = ["Bollinger Bands (20, 2σ)", "RSI (14)", "MACD (12, 26, 9)"] + (["Volume"] if asset == 'Bitcoin' else [])
overlays = st.sidebar.multiselect("Chart overlays", overlay_options, default=[],
                                  help="Indicator families the model's features are built from. Futures volume is not offered: "
                                       "Yahoo's front-month volume is unusable and is not a model input.")
show_bb = "Bollinger Bands (20, 2σ)" in overlays
show_rsi = "RSI (14)" in overlays
show_macd = "MACD (12, 26, 9)" in overlays
show_vol = "Volume" in overlays

st.sidebar.markdown("---")
if st.sidebar.button("Sync live market data", width='stretch',
                     help="Downloads the latest prices from Yahoo Finance into data/raw_live/. The frozen evaluation data is never changed."):
    with st.spinner("Fetching latest data from Yahoo Finance…"):
        ok = update_live_data(asset, refresh_external=True)
    if ok:
        st.cache_data.clear(); st.sidebar.success("Live data synced.")
    else:
        st.sidebar.warning("Yahoo Finance did not return data (rate limit). Using the stored dataset.")

df = load_features(asset)
te_all = load_csv('final_test_results.csv')
cv_all = load_csv('cv_results.csv')
hist = load_test_predictions(asset)
comb_test = test_row(te_all, asset, COMBINED)
naive_test = test_row(te_all, asset, 'Naive')

st.sidebar.markdown("---")
st.sidebar.caption(f"**Served forecast:** Combined ({len(members)} models)  \n**Best single model by validation:** {cv_best or 'n/a'}  \n"
                   f"**Horizon:** {PREDICTION_HORIZON_DAYS} trading day  \n"
                   f"**Data:** Yahoo Finance · {df.attrs.get('source', '—') if df is not None else '—'} · last complete bar "
                   f"{df['timestamp'].iloc[-1].date() if df is not None else '—'}")

# ─────────────────────────────────────────────── title + overview strip
st.title(f"{asset} — Next-Day Price Forecast")
tm = status.get('combined_test') or {}; tn = status.get('test_naive') or {}
if tm and tn:
    da_txt = f"{tm['DirAcc_pct']:.1f}% (chance = 50%, p = {tm['DirAcc_pvalue']:.2f})"
    st.markdown(
        f"<div class='overview'>"
        f"<b>What is predicted:</b> the next trading day's close of {ASSET_CONFIG[asset]['ticker']} (via the log return "
        f"ln(P<sub>t+1</sub>/P<sub>t</sub>)) from {len(status.get('features', []))} backward-looking features. &nbsp;·&nbsp; "
        f"<b>Served forecast:</b> the combination of {len(members)} trained models ({', '.join(members)}), equal weights. &nbsp;·&nbsp; "
        f"<b>Reliability on {tm['n_test']} unseen days</b> ({status['test_period'][0]} → {status['test_period'][1]}): "
        f"direction correct {da_txt}; average price error {tm['MAPE_usd']:.2f}% vs {tn['MAPE_usd']:.2f}% for the random walk "
        f"(\"tomorrow = today\"). <b>The forecast does not beat the random walk</b> (Diebold–Mariano p = {tm['DM_pvalue']:.2f})."
        f"</div>", unsafe_allow_html=True)
else:
    st.info("No evaluation found. Run `make evaluate` (python src/evaluation/backtesting.py) first.")

tab_fc, tab_demo, tab_hist, tab_perf, tab_meth = st.tabs(
    ["Forecast", "Predict a Day (unseen test)", "Test-Set History", "Model Performance", "Methodology"])

# ═══════════════════════════════════════════════ 1. FORECAST
with tab_fc:
    if df is None or len(df) < 2:
        st.info("No processed data found. Run `make preprocess` first.")
    else:
        latest_date = df['timestamp'].iloc[-1].strftime("%Y-%m-%d")
        latest_price, prev_price = df['price'].iloc[-1], df['price'].iloc[-2]
        st.subheader("Latest market data")
        if df.attrs.get('source') == 'live download' and ASSET_CONFIG[asset]['type'] == 'commodity':
            st.caption("Live data: Yahoo's continuous futures series (GC=F / SI=F) follows the current front-month contract, so recent closes can "
                       "differ from the frozen evaluation dataset after a contract roll. The frozen dataset behind every test result is unchanged.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Last complete close ({latest_date})", usd(latest_price),
                  f"{latest_price - prev_price:+,.2f} ({(latest_price / prev_price - 1) * 100:+.2f}%) vs previous day")
        c2.metric("30-day high", usd(df['price'].tail(30).max()))
        c3.metric("30-day low", usd(df['price'].tail(30).min()))
        c4.metric("Daily volatility (30-day σ)", f"{df['log_return'].tail(30).std() * 100:.2f}%",
                  help="Standard deviation of daily log returns over the last 30 rows — the natural scale of any next-day error.")

        # ── prediction
        st.subheader("Next-day prediction")
        if not st.session_state['run_prediction'] or not members:
            st.caption("Press **Run prediction** in the sidebar. Every trained model runs on the latest complete bar and their "
                       "combination is shown here.")
        else:
            try:
                if st.session_state.get('run_result', (None,))[0] != asset:
                    with st.spinner("Checking for the latest complete bar…"):
                        refreshed = False
                        live_path = os.path.join(PROCESSED_DATA_DIR, f'{get_prefix(asset)}_live_features.csv')
                        fresh = os.path.exists(live_path) and (time.time() - os.path.getmtime(live_path)) < 30 * 60
                        if not (live_data_is_current(asset) or fresh):        # download at most once per 30 min while the vendor lags
                            refreshed = update_live_data(asset, refresh_external=True)
                            if refreshed:
                                st.cache_data.clear()
                    with st.spinner(f"Running {len(members)} models…"):
                        r = predict_next_day(asset, COMBINED)
                    st.session_state['run_result'] = (asset, r, refreshed)
                    if refreshed:
                        st.rerun()
                _, r, _ = st.session_state['run_result']
                target = r['target_date']
                up = r['direction'] == 'UP'
                if not live_data_is_current(asset):
                    st.warning(f"Yahoo Finance has not yet published the {asset} bar for "
                               f"{expected_last_complete_bar(ASSET_CONFIG[asset]['type'])} (Bitcoin's previous-day bar often appears a few hours "
                               f"after 00:00 UTC). The forecast below therefore starts from {r['as_of_date']}, the last complete bar available; "
                               f"press Run prediction again later for the newer bar.")
                st.markdown(
                    f"<div class='final'><div class='lbl'>Final prediction — combination of {len(r['individual'])} models · "
                    f"close on {r['as_of_date']} {usd(r['current_price'])} → predicted close for {target}</div>"
                    f"<span class='big'>{usd(r['predicted_price'])}</span> &nbsp; "
                    f"<span class='big {'up' if up else 'down'}'>{'UP ▲' if up else 'DOWN ▼'}</span> &nbsp; "
                    f"<span style='color:#D6D9DE;font-size:1.1rem'>{r['predicted_return_pct']:+.2f}% "
                    f"({r['predicted_price'] - r['current_price']:+,.2f} USD)</span></div>", unsafe_allow_html=True)
                band = r.get('uncertainty_band')
                if band:
                    st.caption(f"±1 RMSE band from the unseen test period: **{usd_md(band[0])} – {usd_md(band[1])}**. The band is far wider than "
                               f"the predicted move — that is the honest picture of next-day uncertainty.")

                # each model's own prediction and the combination
                t = pd.DataFrame(r['individual'])
                t = pd.concat([t, pd.DataFrame([{'model': 'Combined (served)', 'predicted_price': r['predicted_price'],
                                                 'predicted_return_pct': r['predicted_return_pct'], 'direction': r['direction']}])], ignore_index=True)
                t['Predicted close'] = t['predicted_price'].map(usd)
                t['Change'] = t['predicted_return_pct'].map(lambda v: f"{v:+.2f}%")
                t['Direction'] = t['direction'].map({'UP': 'UP ▲', 'DOWN': 'DOWN ▼'})
                st.markdown("**What each model predicted** (the served forecast is the equal-weight mean of the six predicted returns)")
                st.dataframe(t[['model', 'Predicted close', 'Change', 'Direction']].rename(columns={'model': 'Model'}),
                             width='stretch', hide_index=True)
                n_up = sum(m['direction'] == 'UP' for m in r['individual'])
                st.caption(f"{n_up} of {len(r['individual'])} models say UP. The models agree to within a fraction of a percent because "
                           "daily returns are mostly noise: every model has learned to stay close to \"no change\".")

                # forecast chart: last 60 days + next day
                tail = df.tail(60)
                next_date = pd.Timestamp(target)
                fh = go.Figure()
                fh.add_trace(go.Scatter(x=tail['timestamp'], y=tail['price'], mode='lines', name='close',
                                        line=dict(color=COLOR[asset], width=2)))
                if band:
                    fh.add_trace(go.Scatter(x=[next_date, next_date], y=band, mode='lines', name='±1 RMSE band',
                                            line=dict(color=BAND, width=8)))
                fh.add_trace(go.Scatter(x=[tail['timestamp'].iloc[-1], next_date], y=[r['current_price'], r['predicted_price']],
                                        mode='lines+markers', name='combined forecast',
                                        line=dict(color=ACCENT, width=2, dash='dash'), marker=dict(size=9)))
                fh.update_yaxes(tickprefix="$")
                st.plotly_chart(layout(fh, 360, "Last 60 days and the next-day forecast"), use_container_width=True)

                # reliability of the served forecast (from the single test-set evaluation)
                if comb_test is not None and naive_test is not None:
                    st.markdown(f"**How reliable is this forecast?** Single evaluation on {int(comb_test['n_test'])} unseen days "
                                f"({status['test_period'][0]} → {status['test_period'][1]}), side by side with the random walk (\"tomorrow = today\").")
                    beats = comb_test['DM_pvalue'] < 0.05 and comb_test['RMSE_ret'] < naive_test['RMSE_ret']
                    rel_tbl = pd.DataFrame({
                        '': ['Combined forecast', 'Random walk'],
                        'Mean abs. error': [usd(comb_test['MAE_usd']), usd(naive_test['MAE_usd'])],
                        'RMSE': [usd(comb_test['RMSE_usd']), usd(naive_test['RMSE_usd'])],
                        'MAPE': [pct(comb_test['MAPE_usd'], 2), pct(naive_test['MAPE_usd'], 2)],
                        'Direction correct': [f"{pct(comb_test['DirAcc_pct'])} (p = {comb_test['DirAcc_pvalue']:.2f} vs 50%)", '— (no direction)'],
                        'Beats random walk?': [f"{'Yes' if beats else 'No'} (Diebold–Mariano p = {comb_test['DM_pvalue']:.2f})", '—']})
                    st.dataframe(rel_tbl, width='stretch', hide_index=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                logger.exception("Dashboard prediction error")

        # ── price history with the frozen split
        st.subheader("Price history and the frozen train / validation / test split")
        n_rows, heights = 1, [0.6]
        for flag, h in ((show_vol, 0.14), (show_rsi, 0.13), (show_macd, 0.13)):
            if flag:
                n_rows += 1; heights.append(h)
        if n_rows == 1:
            heights = [1.0]
        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=heights)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'], mode='lines', name=f'{asset} close',
                                 line=dict(color=COLOR[asset], width=1.8)), row=1, col=1)
        t_end, v_end = pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END)
        for x0, x1, c in ((df['timestamp'].min(), t_end, '#4CAF50'), (t_end, v_end, '#FFC107'), (v_end, df['timestamp'].max(), '#F44336')):
            fig.add_vrect(x0=x0, x1=x1, fillcolor=c, opacity=0.07, line_width=0, row=1, col=1)
        for x in (t_end, v_end):
            fig.add_vline(x=x, line=dict(color='rgba(255,255,255,0.35)', width=1, dash='dot'), row=1, col=1)
        if show_bb:
            for col, nm, dash in (('BB_Upper', 'BB upper', 'dash'), ('BB_Lower', 'BB lower', 'dash'), ('BB_Mid', 'BB mid (SMA20)', 'dot')):
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], mode='lines', name=nm, line=dict(color='gray', width=1, dash=dash)), row=1, col=1)
        row = 1
        if show_vol and 'volume' in df.columns:
            row += 1
            colors = np.where(df['price'].diff().fillna(0) >= 0, '#4CAF50', '#F44336')
            fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color=colors, opacity=0.6), row=row, col=1)
        if show_rsi:
            row += 1
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], mode='lines', name='RSI (14)', line=dict(color='#E91E63', width=1.2)), row=row, col=1)
            fig.add_hline(y=70, line=dict(color='red', width=1, dash='dot'), row=row, col=1)
            fig.add_hline(y=30, line=dict(color='green', width=1, dash='dot'), row=row, col=1)
        if show_macd:
            row += 1
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MACD'], mode='lines', name='MACD', line=dict(color='#2196F3', width=1.2)), row=row, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MACD_Signal'], mode='lines', name='Signal', line=dict(color='#FF9800', width=1.2)), row=row, col=1)
            h_ = df['MACD'] - df['MACD_Signal']
            fig.add_trace(go.Bar(x=df['timestamp'], y=h_, name='Histogram', marker_color=np.where(h_ >= 0, '#4CAF50', '#F44336')), row=row, col=1)
        fig.update_yaxes(title_text="Price (USD)", tickprefix="$", row=1, col=1)
        fig.update_xaxes(rangeselector=dict(buttons=[dict(count=6, label="6M", step="month", stepmode="backward"),
                                                     dict(count=1, label="1Y", step="year", stepmode="backward"),
                                                     dict(count=2, label="2Y", step="year", stepmode="backward"),
                                                     dict(label="All", step="all")],
                                            bgcolor="#262C38", activecolor="#3A4252", font=dict(color="white")), row=1, col=1)
        st.plotly_chart(layout(fig, 460 if n_rows == 1 else 460 + (n_rows - 1) * 110), use_container_width=True)
        st.caption(f"Shading: green = training (≤ {TRAIN_END}) · amber = validation (≤ {VAL_END}) · red = unseen test (evaluated once). "
                   "Overlays are the indicator families the model's features are built from; the model receives them as scale-free ratios.")

# ═══════════════════════════════════════════════ 2. PREDICT A DAY (unseen test period)
with tab_demo:
    st.subheader("Predict a day of the unseen test period")
    st.markdown(f"Stand on any day after **{VAL_END}** — a period never used for training, validation or model selection. "
                f"All models see data **only up to that day**, their combined forecast for the next trading day is shown, and the actual close is then revealed.")
    if hist is None or df is None:
        st.warning("Run `make evaluate` first to generate the unseen-test predictions.")
    else:
        test_days = list(hist['date'].dt.date)
        c1, c2 = st.columns([3, 1])
        with c1:
            pick = st.selectbox("Stand on this day (data available up to and including it)", test_days, index=len(test_days) - 1,
                                format_func=lambda d: d.strftime('%Y-%m-%d (%a)'))
        with c2:
            st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
            go_demo = st.button("Generate prediction", type="primary", width='stretch')
        if go_demo or st.session_state.get('demo_done'):
            st.session_state['demo_done'] = True
            try:
                r = predict_for_date(asset, str(pick), COMBINED)
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric(f"Close on {r['as_of_date']}", usd(r['current_price']))
                m2.metric(f"Predicted close for {r['actual_date'] or 'next day'}", usd(r['predicted_price']), f"{r['predicted_return_pct']:+.2f}%")
                if r['actual_price'] is not None:
                    m3.metric("Actual close", usd(r['actual_price']), f"{r['actual_return_pct']:+.2f}%")
                    m4.metric("Prediction error", f"{r['error_pct']:+.2f}%", f"{r['error_usd']:+,.2f} USD", delta_color="off")
                    hit = r['direction_hit']
                    m5.metric("Direction", f"{r['direction']} → {'HIT ✔' if hit else ('FAIL ✘' if hit is not None else 'flat')}")
                else:
                    m3.metric("Actual close", "not yet known")
                st.caption(f"Forecast: **combined** of {len(r['individual'])} models · horizon {r['horizon_days']} trading day · "
                           f"{'inside the unseen test period' if r['in_unseen_test_period'] else 'inside the training/validation period'} · "
                           "the prediction uses only data up to the chosen day (frozen dataset).")

                col_pred = f'pred_close_{COMBINED}'
                if col_pred in hist.columns:
                    fd = go.Figure()
                    fd.add_trace(go.Scatter(x=hist['target_date'], y=hist['actual_close'], mode='lines', name='actual close',
                                            line=dict(color=COLOR[asset], width=2)))
                    fd.add_trace(go.Scatter(x=hist['target_date'], y=hist[col_pred], mode='lines', name='combined predicted close',
                                            line=dict(color=ACCENT, width=1.4, dash='dash')))
                    if r['actual_price'] is not None:
                        fd.add_trace(go.Scatter(x=[pd.Timestamp(r['actual_date'])], y=[r['predicted_price']], mode='markers', name='this prediction',
                                                marker=dict(color=ACCENT, size=13, symbol='diamond', line=dict(color='white', width=1))))
                        fd.add_trace(go.Scatter(x=[pd.Timestamp(r['actual_date'])], y=[r['actual_price']], mode='markers', name='actual',
                                                marker=dict(color=COLOR[asset], size=13, symbol='circle', line=dict(color='white', width=1))))
                    fd.update_yaxes(tickprefix="$")
                    st.plotly_chart(layout(fd, 400, f"Unseen test period {hist['target_date'].min().date()} → {hist['target_date'].max().date()}: "
                                                    f"actual vs predicted close"), use_container_width=True)
                    st.caption("A predicted-price line always hugs the actual line because it starts from today's close. "
                               "Whether the models have skill is only visible in return space — see Model Performance.")

                if r['individual']:
                    t = pd.DataFrame(r['individual'])
                    t['Predicted'] = t['predicted_price'].map(usd)
                    t['Change'] = t['predicted_return_pct'].map(lambda v: f"{v:+.2f}%")
                    t['Error'] = t['error_pct'].map(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
                    t['Direction'] = t['direction_hit'].map(lambda v: 'HIT ✔' if v is True else ('FAIL ✘' if v is False else '—'))
                    st.markdown("**Each model on this day** (averaged into the combined forecast above)")
                    st.dataframe(t[['model', 'Predicted', 'Change', 'Error', 'Direction']].rename(columns={'model': 'Model'}),
                                 width='stretch', hide_index=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                logger.exception("Demo prediction error")

# ═══════════════════════════════════════════════ 3. TEST-SET HISTORY
with tab_hist:
    st.subheader("Test-set history — every unseen day, combined forecast")
    if hist is None or f'pred_close_{COMBINED}' not in hist.columns:
        st.warning("Run `make evaluate` first.")
    else:
        m = COMBINED
        e_m, e_n = hist[f'error_pct_{m}'].abs(), hist['error_pct_Naive'].abs()
        hits = int(hist[f'direction_hit_{m}'].sum())
        st.caption(f"{len(hist)} unseen days, {hist['target_date'].min().date()} → {hist['target_date'].max().date()}. Summary of the table below:")
        st.dataframe(pd.DataFrame({'': ['Combined forecast', 'Random walk'],
                                   'Mean abs. error (% of price)': [f"{e_m.mean():.2f}%", f"{e_n.mean():.2f}%"],
                                   'Median abs. error (% of price)': [f"{e_m.median():.2f}%", f"{e_n.median():.2f}%"],
                                   'Largest miss': [f"{e_m.max():.2f}%", f"{e_n.max():.2f}%"],
                                   'Direction correct': [f"{hits / len(hist) * 100:.1f}% ({hits} of {len(hist)} days; chance ≈ 50%)", '— (no direction)']}),
                     width='stretch', hide_index=True)
        h = pd.DataFrame({'Predicted on': hist['date'].dt.date, 'For': hist['target_date'].dt.date,
                          'Predicted ($)': hist[f'pred_close_{m}'].round(2), 'Actual ($)': hist['actual_close'].round(2),
                          'Error ($)': (hist[f'pred_close_{m}'] - hist['actual_close']).round(2),
                          'Error (%)': hist[f'error_pct_{m}'].round(2),
                          'Random walk error (%)': hist['error_pct_Naive'].round(2),
                          'Direction': hist[f'direction_hit_{m}'].map({1: 'HIT', 0: 'FAIL'})})
        st.dataframe(h.sort_values('For', ascending=False), width='stretch', hide_index=True, height=440,
                     column_config={'Predicted ($)': st.column_config.NumberColumn(format="$%.2f"),
                                    'Actual ($)': st.column_config.NumberColumn(format="$%.2f"),
                                    'Error ($)': st.column_config.NumberColumn(format="%+.2f"),
                                    'Error (%)': st.column_config.NumberColumn(format="%+.2f%%"),
                                    'Random walk error (%)': st.column_config.NumberColumn(format="%+.2f%%")})
        st.download_button("Download test-set history (CSV)", h.to_csv(index=False).encode(), file_name=f"{asset.lower()}_combined_test_history.csv")
        st.caption(f"Every row is an out-of-sample prediction: the models were trained on data up to {VAL_END} and each prediction used "
                   "only data up to the 'Predicted on' date. The random-walk error is the error of predicting \"tomorrow = today\".")

# ═══════════════════════════════════════════════ 4. MODEL PERFORMANCE
with tab_perf:
    if cv_all is None or te_all is None or not tm:
        st.warning("Run `make evaluate` to generate results.")
    else:
        s = status
        st.subheader("Unseen test set — evaluated once")
        st.caption(f"Test period {s['test_period'][0]} → {s['test_period'][1]} ({tm['n_test']} days). The served forecast is the **Combined** row "
                   f"(equal-weight mean of the six trained models — no fitted weights, so nothing was chosen on the test set). "
                   f"The best single model by walk-forward validation is **{cv_best}**.")
        t = te_all[te_all['asset'] == asset].sort_values('RMSE_ret')
        is_naive = (t['model'] == 'Naive').values
        label = lambda m_: f"{m_} (served)" if m_ == COMBINED else (f"{m_} (best single by CV)" if m_ == cv_best else m_)
        show = pd.DataFrame({
            'Model': t['model'].map(label),
            'MAE ($)': t['MAE_usd'].map(usd), 'RMSE ($)': t['RMSE_usd'].map(usd), 'MAPE': t['MAPE_usd'].map(lambda v: pct(v, 2)),
            'RMSE (return)': t['RMSE_ret'].map(lambda v: f"{v:.5f}"), 'R² (return)': t['R2_ret'].map(lambda v: f"{v:+.3f}"),
            'Direction correct': np.where(is_naive, '—', t['DirAcc_pct'].map(lambda v: pct(v)) + ' (p ' + t['DirAcc_pvalue'].map(lambda v: f"{v:.2f}") + ')'),
            'DM p vs random walk': np.where(is_naive, '—', t['DM_pvalue'].map(lambda v: f"{v:.2f}")),
            'Long/flat strategy': t['strategy_return_pct'].map(lambda v: f"{v:+.1f}%"),
            'Buy & hold': t['buy_hold_return_pct'].map(lambda v: f"{v:+.1f}%")})
        st.dataframe(show, width='stretch', hide_index=True)

        naive_r, best = t[is_naive].iloc[0], t[t['model'] == COMBINED].iloc[0]
        rel = (best['RMSE_ret'] / naive_r['RMSE_ret'] - 1) * 100
        if best['DM_pvalue'] < 0.05 and rel < 0:
            verdict = f"it beats the random walk on magnitude ({rel:+.2f}%, Diebold–Mariano p = {best['DM_pvalue']:.2f}, significant at 5%)"
        elif abs(rel) < 0.05:
            verdict = f"it is indistinguishable from the random walk (Diebold–Mariano p = {best['DM_pvalue']:.2f})"
        else:
            verdict = (f"it is {abs(rel):.2f}% {'below' if rel < 0 else 'above'} the random walk's error — "
                       f"not a significant difference (Diebold–Mariano p = {best['DM_pvalue']:.2f})")
        da_sig = 'significant' if best['DirAcc_pvalue'] < 0.05 else 'not significant'
        st.info(f"**Honest reading.** On the unseen test set the combined forecast's RMSE (return) is {best['RMSE_ret']:.5f} vs {naive_r['RMSE_ret']:.5f} "
                f"for the random walk — {verdict}. Direction correct on {best['DirAcc_pct']:.1f}% of {int(best['DirAcc_n'])} days "
                f"(binomial p = {best['DirAcc_pvalue']:.2f}, {da_sig}). MAPE is not accuracy: \"tomorrow = today\" scores the same MAPE. "
                "R² on price levels is intentionally not shown — a random walk scores ≈ 0.95 there.")

        # comparison chart: every model vs the random walk
        tt = t.copy(); tt['label'] = tt['model'].map(lambda m_: f"{m_}*" if m_ == COMBINED else m_)
        fc = make_subplots(rows=1, cols=2, subplot_titles=("RMSE of the next-day return (lower is better)", "Direction correct, % (higher is better)"),
                           horizontal_spacing=0.08)
        colors = [NAIVE if m_ == 'Naive' else (ACCENT if m_ == COMBINED else '#4E7FA8') for m_ in tt['model']]
        fc.add_trace(go.Bar(x=tt['label'], y=tt['RMSE_ret'], marker_color=colors, name='RMSE (return)', showlegend=False), row=1, col=1)
        fc.add_hline(y=float(naive_r['RMSE_ret']), line=dict(color=NAIVE, dash='dash', width=1.5), row=1, col=1,
                     annotation_text="random walk", annotation_position="top left", annotation_font_color=NAIVE)
        nn = tt[tt['model'] != 'Naive']
        fc.add_trace(go.Bar(x=nn['label'], y=nn['DirAcc_pct'], marker_color=[ACCENT if m_ == COMBINED else '#4E7FA8' for m_ in nn['model']],
                            name='direction %', showlegend=False), row=1, col=2)
        fc.add_hline(y=50, line=dict(color=NAIVE, dash='dash', width=1.5), row=1, col=2, annotation_text="chance (50%)",
                     annotation_position="top left", annotation_font_color=NAIVE)
        fc.update_yaxes(range=[float(tt['RMSE_ret'].min()) * 0.985, float(tt['RMSE_ret'].max()) * 1.01], row=1, col=1)
        fc.update_yaxes(range=[40, 65], row=1, col=2)
        fc.update_layout(hovermode="closest")
        st.plotly_chart(layout(fc, 340, legend=False), use_container_width=True)
        st.caption("The bar marked with an asterisk is the served combined forecast. The RMSE axis is zoomed: all models lie within ±1% of the random walk.")

        st.subheader("Walk-forward validation — the model-selection evidence")
        c = cv_all[cv_all['asset'] == asset].sort_values('RMSE_ret_mean')
        showc = pd.DataFrame({'Model': c['model'].map(label),
                              'RMSE (return) mean ± std over 4 folds': c['RMSE_ret_mean'].map(lambda v: f"{v:.5f}") + ' ± ' + c['RMSE_ret_std'].map(lambda v: f"{v:.5f}"),
                              'MAE (return)': c['MAE_ret_mean'].map(lambda v: f"{v:.5f}"), 'R² (return)': c['R2_ret_mean'].map(lambda v: f"{v:+.3f}"),
                              'Direction correct': np.where(c['model'] == 'Naive', '—', c['DirAcc_pct_mean'].map(lambda v: pct(v))),
                              'RMSE ($)': c['RMSE_usd_mean'].map(usd)})
        st.dataframe(showc, width='stretch', hide_index=True)
        st.caption("Four expanding-window folds inside the train+validation period (the test set is never used here). The Combined row averages the "
                   "members' out-of-fold predictions. All models lie within ~0.5% of each other and of the random walk, so the ranking is not "
                   "statistically decisive — which is why the served forecast is the plain equal-weight combination rather than a single 'winner'.")

        # predicted vs actual in return space for the combined forecast
        if hist is not None and f'pred_return_{COMBINED}' in hist.columns:
            st.subheader("Predicted vs actual next-day returns — combined forecast on the unseen test period")
            pr, ar = hist[f'pred_return_{COMBINED}'] * 100, hist['actual_return'] * 100
            f1, f2 = st.columns([3, 2])
            with f1:
                fr = go.Figure()
                fr.add_trace(go.Scatter(x=hist['target_date'], y=ar, name='actual return', line=dict(color='rgba(255,255,255,0.75)', width=1)))
                fr.add_trace(go.Scatter(x=hist['target_date'], y=pr, name='combined prediction', line=dict(color=ACCENT, width=1.8)))
                fr.update_yaxes(ticksuffix="%")
                st.plotly_chart(layout(fr, 340, "Daily returns: actual vs predicted (%)"), use_container_width=True)
            with f2:
                sc = go.Figure()
                sc.add_trace(go.Scatter(x=ar, y=pr, mode='markers', marker=dict(color=ACCENT, size=6, opacity=0.7), name='days'))
                lim = float(np.abs(ar).max()) * 1.05
                sc.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode='lines', line=dict(color='red', dash='dash', width=1), name='perfect prediction'))
                sc.update_xaxes(title_text="actual (%)", range=[-lim, lim]); sc.update_yaxes(title_text="predicted (%)", range=[-lim, lim])
                sc.update_layout(hovermode="closest")
                st.plotly_chart(layout(sc, 340, "Scatter (equal axes)"), use_container_width=True)
            if comb_test is not None:
                st.caption(f"The prediction line barely moves (std {comb_test['pred_std'] * 100:.2f}% vs {comb_test['true_std'] * 100:.2f}% for actual returns): "
                           "the models have learned that tomorrow's move is mostly unpredictable and stay close to \"no change\".")

        # feature importance (tree / linear members)
        fi = load_csv('feature_importance.csv')
        if fi is not None:
            fi_models = [m_ for m_ in ('CatBoost', 'LightGBM', 'RandomForest', 'Ridge') if len(fi[(fi['asset'] == asset) & (fi['model'] == m_)])]
            if fi_models:
                pick_m = st.selectbox("Which inputs does a member model rely on?", fi_models, index=0,
                                      help="GRU / LSTM do not expose feature importance; the combined forecast averages all six.")
                f_sel = fi[(fi['asset'] == asset) & (fi['model'] == pick_m)].sort_values('importance', ascending=False)
                top = f_sel.head(15).iloc[::-1]
                fb = go.Figure(go.Bar(x=top['importance'], y=top['feature'], orientation='h', marker_color=ACCENT))
                fb.update_layout(hovermode="closest")
                st.plotly_chart(layout(fb, 420, f"{pick_m}: top 15 of {len(f_sel)} features (normalised importance)", legend=False), use_container_width=True)

        reg = load_csv('regime_analysis.csv')
        if reg is not None:
            with st.expander("Performance by market regime (combined forecast vs random walk, unseen test period)"):
                rg = reg[(reg['asset'] == asset) & (reg['regime_type'] != 'day_regime')]
                st.dataframe(pd.DataFrame({'Regime': rg['regime'], 'Days': rg['n_days'],
                                           'MAE % (combined)': rg['mae_pct_served'].map(lambda v: f"{v:.2f}"),
                                           'MAE % (random walk)': rg['mae_pct_naive'].map(lambda v: f"{v:.2f}"),
                                           'Direction correct (combined)': rg['dir_hit_served_pct'].map(lambda v: pct(v))}),
                             width='stretch', hide_index=True)
                st.caption("Regimes are defined on the day the prediction is made (30-day volatility tercile; sign of the 20-day return), so they never use future information.")
        with st.expander("Design experiments (validation only): data size, feature ablation, horizon, volatility target"):
            for fname, title in (('E1_data_size.csv', 'E1 — training-data size (fixed validation window)'),
                                 ('E2_feature_ablation.csv', 'E2 — feature-group ablation (walk-forward CV)'),
                                 ('E3_horizon.csv', 'E3 — next-day vs 5-day return target'),
                                 ('E4_volatility.csv', 'E4 — 22-day realised-volatility target')):
                ex = load_csv(os.path.join('experiments', fname))
                if ex is not None:
                    st.markdown(f"**{title}**")
                    st.dataframe(ex[ex['asset'] == asset].round(4), width='stretch', hide_index=True)
        with st.expander("Training diagnostics: over-fitting gap and GRU/LSTM loss curves"):
            for name in ('overfitting_gap.png', f'{get_prefix(asset)}_loss_curves.png'):
                p = os.path.join(FIGURES_DIR, name)
                if os.path.exists(p):
                    st.image(p, width='stretch')

# ═══════════════════════════════════════════════ 5. METHODOLOGY
with tab_meth:
    st.subheader("Methodology")
    close_rule = ("Gold and Silver close at the 13:30 ET COMEX settlement, before the US equity, rates and FX closes, so their macro returns and the "
                  "High/Low-based features (hl_range, ATR/price, ADX) are the *previous* session's values — only information known when the price is fixed is used."
                  if asset != 'Bitcoin' else
                  "BTC-USD closes at 00:00 UTC, after every US market close, so same-day macro values are known at the close.")
    st.markdown(f"""
**Target.** The next trading day's log return `y_t = ln(P_(t+1) / P_t)`, converted to a price with `P̂_(t+1) = P_t · exp(ŷ_t)`.
Predicting the return (not the price) makes the target stationary and makes the random-walk baseline (ŷ = 0) explicit.

**Data.** Yahoo Finance daily data from {status.get('data_start', '2018')} ({status.get('n_train', '?')} training days, {status.get('n_val', '?')} validation days).
Only **complete** daily bars are used (the running bar of the current day is dropped at download time).
**Split** — frozen, chronological, no shuffling: train ≤ {TRAIN_END} · validation ≤ {VAL_END} · unseen test = remainder.
The scaler is fitted on train only; hyper-parameters are chosen by 4-fold expanding-window walk-forward validation inside train+val; the test set is evaluated once.

**Features ({len(status.get('features', []))}).** Stationary and backward-looking: today's and lagged log returns, 20-day momentum, rolling / HAR / EWMA volatility,
RSI, ADX, ROC, MACD/price, price/EMA ratios, Bollinger %B and width, ATR/price, high–low range, macro daily returns
(S&P 500, VIX{', DXY, oil, 10-year yield' if asset != 'Bitcoin' else ', Fear & Greed, day-of-week, volume change'}){', gold return' if asset == 'Silver' else ''}.
Raw price levels are shown on charts but never fed to a model. **Close-time rule.** {close_rule}

**Models.** Six trained models — Ridge, Random Forest, LightGBM, CatBoost, GRU, LSTM — plus baselines (random walk, historical mean, ARIMA) and a stacked
ensemble experiment, all scored on identical days. Models with a stopping rule (trees / epochs) use validation for it in phase 1 and are refit on train+val in phase 2.

**Served forecast — the combination.** *Run prediction* runs all six models on the same input window and averages their predicted log returns with
**equal weights**; the sign of the average is the UP/DOWN call. Equal weights are the standard, robust choice for combining forecasts of similar quality
(no weights are fitted, so nothing is selected on the test set), and the combination is evaluated exactly like every single model — on the walk-forward
folds and once on the unseen test set (row *Combined*). The best single model by validation ({cv_best}) is reported for comparison.

**Metrics.** MAE / RMSE / MAPE in USD (what a user sees); RMSE and R² of the return (what is actually predicted); directional accuracy on non-flat days with a
binomial p-value against 50 %; Diebold–Mariano test of squared errors against the random walk; a long/flat strategy backtest with 10 bps cost.
""")
    st.markdown("**Served forecast — details**")
    fit = status.get('fit_info') or {}
    stop = (f"{fit['epochs_used']} epochs" if 'epochs_used' in fit else
            f"{fit.get('iterations_used', fit.get('n_estimators_used', '—'))} boosting rounds" if fit else '—')
    details = pd.DataFrame({'Item': ['Served forecast', 'Members', 'Combination rule',
                                     'Walk-forward RMSE (return) — combined vs random walk', 'Walk-forward direction correct (combined)',
                                     'Best single model by validation', f'{cv_best} stopping point (chosen on validation)', f'{cv_best} hyper-parameters'],
                            'Value': [COMBINED, ', '.join(members) or '—', status.get('combined_rule', '—'),
                                      f"{status.get('combined_cv_rmse_ret') or float('nan'):.5f} vs {status.get('cv_rmse_ret_naive', float('nan')):.5f}",
                                      pct(status.get('combined_cv_dir_acc')),
                                      f"{cv_best or '—'} ({status.get('selection_rule', '—')})", stop,
                                      ', '.join(f"{k} = {v}" for k, v in (status.get('params') or {}).items())]})
    st.dataframe(details, width='stretch', hide_index=True)
    st.caption("Full detail: `docs/METHODOLOGY.md`, `docs/FEATURES.md`, `docs/RESULTS.md`.")

st.markdown(f"<div class='footer'>{DISCLAIMER}</div>", unsafe_allow_html=True)
