"""
Build the final FYP report PDF from the project's real artefacts.

    python docs/build_report.py

Reads: results/*.csv, results/tuning/best_params.json, data/models/model_status.json,
       results/stacking/*.json, results/figures/*.png, docs/figures/*.png
Writes: docs/FYP_Final_Report.pdf, docs/Executive_Summary.pdf
"""
import os
import sys
import json
import datetime as dt
import pandas as pd
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether, NextPageTemplate)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from config import RESULTS_DIR, FIGURES_DIR, TUNING_DIR, MODEL_STATUS_PATH, TRAIN_END, VAL_END, ASSETS, SEQ_LEN, CV_FOLDS

OUT_REPORT = os.path.join(ROOT, 'docs', 'FYP_Final_Report.pdf')
OUT_SUMMARY = os.path.join(ROOT, 'docs', 'Executive_Summary.pdf')
DOC_FIG = os.path.join(ROOT, 'docs', 'figures')

# ------------------------------------------------------------------ fonts
FONT_DIR = '/System/Library/Fonts/Supplemental'
pdfmetrics.registerFont(TTFont('Body', os.path.join(FONT_DIR, 'Arial.ttf')))
pdfmetrics.registerFont(TTFont('Body-Bold', os.path.join(FONT_DIR, 'Arial Bold.ttf')))
pdfmetrics.registerFont(TTFont('Body-Italic', os.path.join(FONT_DIR, 'Arial Italic.ttf')))
pdfmetrics.registerFont(TTFont('Head', os.path.join(FONT_DIR, 'Georgia Bold.ttf')))
pdfmetrics.registerFont(TTFont('Head-Reg', os.path.join(FONT_DIR, 'Georgia.ttf')))
pdfmetrics.registerFont(TTFont('Mono', os.path.join(ROOT, 'venv/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSansMono.ttf')))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily('Body', normal='Body', bold='Body-Bold', italic='Body-Italic', boldItalic='Body-Bold')

NAVY = colors.HexColor('#1F3A5F')
GREY = colors.HexColor('#555555')
LIGHT = colors.HexColor('#F2F4F7')
LINE = colors.HexColor('#C8CDD4')

S = {
    'title': ParagraphStyle('title', fontName='Head', fontSize=24, leading=30, textColor=NAVY, alignment=TA_CENTER),
    'subtitle': ParagraphStyle('subtitle', fontName='Head-Reg', fontSize=13, leading=18, textColor=GREY, alignment=TA_CENTER),
    'cover': ParagraphStyle('cover', fontName='Body', fontSize=11, leading=16, alignment=TA_CENTER),
    'h1': ParagraphStyle('h1', fontName='Head', fontSize=16, leading=20, textColor=NAVY, spaceBefore=14, spaceAfter=6),
    'h2': ParagraphStyle('h2', fontName='Body-Bold', fontSize=11.5, leading=15, textColor=NAVY, spaceBefore=9, spaceAfter=3),
    'body': ParagraphStyle('body', fontName='Body', fontSize=9.8, leading=13.6, alignment=TA_JUSTIFY, spaceAfter=5),
    'bullet': ParagraphStyle('bullet', fontName='Body', fontSize=9.8, leading=13.4, leftIndent=12, bulletIndent=2, spaceAfter=2),
    'small': ParagraphStyle('small', fontName='Body', fontSize=8.3, leading=10.5),
    'smallb': ParagraphStyle('smallb', fontName='Body-Bold', fontSize=8.3, leading=10.5),
    'caption': ParagraphStyle('caption', fontName='Body-Italic', fontSize=8.5, leading=11, textColor=GREY, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10),
    'code': ParagraphStyle('code', fontName='Mono', fontSize=8, leading=10.5, leftIndent=8, backColor=LIGHT, borderPadding=4, spaceAfter=6),
    'box': ParagraphStyle('box', fontName='Body', fontSize=9.6, leading=13.2, backColor=LIGHT, borderPadding=7, borderColor=LINE, borderWidth=0.6, spaceBefore=10, spaceAfter=10),
}

# ------------------------------------------------------------------ data
cv = pd.read_csv(os.path.join(RESULTS_DIR, 'cv_results.csv'))
te = pd.read_csv(os.path.join(RESULTS_DIR, 'final_test_results.csv'))
tv = pd.read_csv(os.path.join(RESULTS_DIR, 'train_val_metrics.csv'))
eda = pd.read_csv(os.path.join(RESULTS_DIR, 'eda_summary.csv'))
status = json.load(open(MODEL_STATUS_PATH))
best = json.load(open(os.path.join(TUNING_DIR, 'best_params.json')))
stack = {a: json.load(open(os.path.join(RESULTS_DIR, 'stacking', f'{a.lower()}_stack_weights.json'))) for a in ASSETS}
tv_piv = tv.pivot_table(index=['asset', 'model'], columns='split', values='RMSE_ret')

ORDER = ['Naive', 'Naive-Mean', 'ARIMA', 'Ridge', 'RandomForest', 'LightGBM', 'CatBoost', 'GRU', 'LSTM', 'Stacked']
NICE = {'Naive': 'Naive (random walk)', 'Naive-Mean': 'Historical mean', 'RandomForest': 'Random Forest', 'Stacked': 'Stacked ensemble'}
BA = pd.read_csv(os.path.join(RESULTS_DIR, 'experiments', 'before_after.csv'))
EXP = {k: pd.read_csv(os.path.join(RESULTS_DIR, 'experiments', f)) for k, f in (('E1', 'E1_data_size.csv'), ('E2', 'E2_feature_ablation.csv'), ('E3', 'E3_horizon.csv'), ('E4', 'E4_volatility.csv'))}
REG = pd.read_csv(os.path.join(RESULTS_DIR, 'regime_analysis.csv'))


def row(asset, model):
    r = te[(te.asset == asset) & (te.model == model)]
    return r.iloc[0] if len(r) else None


def cvrow(asset, model):
    r = cv[(cv.asset == asset) & (cv.model == model)]
    return r.iloc[0] if len(r) else None


def P(text, style='body'):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S['bullet'], bulletText='•') for t in items]


def table(data, col_widths, header=True, font=8.2, align_right_from=1, zebra=True):
    styled = [[Paragraph(str(c), S['smallb'] if (header and i == 0) else S['small']) if not isinstance(c, Paragraph) else c
               for c in r] for i, r in enumerate(data)]
    t = Table(styled, colWidths=col_widths, repeatRows=1 if header else 0)
    st = [('FONTNAME', (0, 0), (-1, -1), 'Body'), ('FONTSIZE', (0, 0), (-1, -1), font),
          ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LINEBELOW', (0, -1), (-1, -1), 0.6, LINE),
          ('TOPPADDING', (0, 0), (-1, -1), 2.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
          ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4)]
    if header:
        st += [('BACKGROUND', (0, 0), (-1, 0), NAVY), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
               ('LINEBELOW', (0, 0), (-1, 0), 0.8, NAVY)]
        for c in range(len(data[0])):
            styled[0][c].style = ParagraphStyle('hdr', parent=S['smallb'], textColor=colors.white)
    if zebra:
        for i in range(1, len(data)):
            if i % 2 == 0:
                st.append(('BACKGROUND', (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(st))
    return t


def fig(path, width_cm, caption, max_height_cm=None):
    im = PILImage.open(path)
    w, h = im.size
    width = width_cm * cm
    height = width * h / w
    if max_height_cm and height > max_height_cm * cm:
        height = max_height_cm * cm
        width = height * w / h
    return KeepTogether([Image(path, width=width, height=height), Paragraph(caption, S['caption'])])


def crop_image(src, dst, box):
    if not os.path.exists(dst):
        PILImage.open(src).crop(box).save(dst)
    return dst


# ------------------------------------------------------------------ page furniture
FOOTER = ['Multi-Asset Next-Day Return Prediction — Final Year Project Report']


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            if self._pageNumber > 1:
                self.setFont('Body', 8); self.setFillColor(GREY)
                self.drawString(2.2 * cm, 1.3 * cm, FOOTER[0])
                self.drawRightString(A4[0] - 2.2 * cm, 1.3 * cm, f'Page {self._pageNumber} of {n}')
                self.setStrokeColor(LINE); self.line(2.2 * cm, 1.7 * cm, A4[0] - 2.2 * cm, 1.7 * cm)
            super().showPage()
        super().save()


def workflow_drawing():
    steps = ['Data\ncollection', 'Cleaning &\nfeatures', 'Frozen\nchrono split', 'Walk-forward\ntuning', 'Two-phase\ntraining',
             'Test\nevaluation', 'Dashboard\n& API']
    d = Drawing(16.4 * cm, 2.6 * cm)
    bw, bh, gap = 2.05 * cm, 1.5 * cm, 0.33 * cm
    x0 = 0.05 * cm
    for i, s in enumerate(steps):
        x = x0 + i * (bw + gap)
        d.add(Rect(x, 0.55 * cm, bw, bh, fillColor=LIGHT, strokeColor=NAVY, strokeWidth=0.8, rx=3, ry=3))
        for j, line in enumerate(s.split('\n')):
            d.add(String(x + bw / 2, 0.55 * cm + bh - 0.55 * cm - j * 0.42 * cm, line, fontName='Body', fontSize=7.6,
                         fillColor=NAVY, textAnchor='middle'))
        if i < len(steps) - 1:
            ax = x + bw
            d.add(Line(ax + 0.03 * cm, 0.55 * cm + bh / 2, ax + gap - 0.12 * cm, 0.55 * cm + bh / 2, strokeColor=NAVY, strokeWidth=0.8))
            d.add(Polygon([ax + gap - 0.14 * cm, 0.55 * cm + bh / 2 + 0.1 * cm, ax + gap - 0.02 * cm, 0.55 * cm + bh / 2,
                           ax + gap - 0.14 * cm, 0.55 * cm + bh / 2 - 0.1 * cm], fillColor=NAVY, strokeColor=NAVY))
    labels = ['yfinance API', 'preprocessing.py', 'config.py dates', 'tune_models.py', 'train_models.py', 'backtesting.py', 'streamlit / fastapi']
    for i, l in enumerate(labels):
        d.add(String(x0 + i * (bw + gap) + bw / 2, 0.2 * cm, l, fontName='Body', fontSize=5.6, fillColor=GREY, textAnchor='middle'))
    return d


# ------------------------------------------------------------------ content helpers
def results_table(asset):
    hdr = ['Model', 'CV RMSE (ret)\nmean ± std', 'Test RMSE\n(ret)', 'Test R²\n(ret)', 'MAE ($)', 'RMSE ($)', 'MAPE %', 'Dir. acc. %\n(p-value)', 'DM p vs\nnaive']
    data = [hdr]
    served = status[asset]['primary_model']
    for m in ORDER:
        r, c = row(asset, m), cvrow(asset, m)
        if r is None:
            continue
        name = NICE.get(m, m) + (' (served)' if m == served else '')
        cvtxt = f"{c['RMSE_ret_mean']:.5f} ± {c['RMSE_ret_std']:.5f}" if c is not None else '—'
        da = '—' if m == 'Naive' else f"{r['DirAcc_pct']:.1f} ({r['DirAcc_pvalue']:.2f})"
        dm = '—' if m == 'Naive' else f"{r['DM_pvalue']:.2f}"
        data.append([name, cvtxt, f"{r['RMSE_ret']:.5f}", f"{r['R2_ret']:+.3f}", f"{r['MAE_usd']:,.2f}", f"{r['RMSE_usd']:,.2f}",
                     f"{r['MAPE_usd']:.2f}", da, dm])
    t = table(data, [3.5 * cm, 2.9 * cm, 1.7 * cm, 1.5 * cm, 1.6 * cm, 1.7 * cm, 1.3 * cm, 1.9 * cm, 1.3 * cm], font=7.6)
    t.setStyle(TableStyle([('ALIGN', (1, 1), (-1, -1), 'RIGHT')]))
    return t


def exec_summary_flowables():
    f = []
    f.append(P('Executive Summary', 'h1'))
    f.append(P(
        'This project is a machine-learning system that forecasts the <b>next trading day\'s return</b> of three assets — '
        'Bitcoin (BTC-USD), Gold (GC=F futures) and Silver (SI=F futures) — and presents the forecast, together with its '
        'measured reliability, in a web dashboard and a REST API. The forecast target is the next-day log return '
        '<i>y<sub>t</sub> = ln(P<sub>t+1</sub>/P<sub>t</sub>)</i>; the price shown to the user is recovered as '
        '<i>P<sub>t</sub>·exp(ŷ)</i>. Daily data from Yahoo Finance (2018-01-01 to 2026-09-12; Bitcoin 3,147 usable days, '
        'Gold/Silver 2,156 exchange days) is enriched with macro series (US Dollar Index, WTI crude, 10-year Treasury yield, '
        'S&amp;P 500, VIX) and the crypto Fear &amp; Greed index.'))
    f.append(P(
        f'<b>Methodology.</b> A frozen chronological split (train ≤ {TRAIN_END}, validation ≤ {VAL_END}, test = '
        f'2026-02-18 → 2026-09-12) with no shuffling; a scaler fitted on training rows only; 28–29 stationary, backward-looking '
        f'features; hyper-parameters and the served model chosen by {CV_FOLDS}-fold expanding-window walk-forward validation inside '
        'train+val; and a single evaluation of the untouched test set. Ten model families are compared on identical days: a '
        'random-walk baseline, historical mean, ARIMA, Ridge, Random Forest, LightGBM, CatBoost, GRU, LSTM and a stacked ensemble. '
        'Every feature obeys a <i>close-time rule</i> — it may only use what is known when the asset\'s daily close is fixed (same-day '
        'macro data for Bitcoin, previous-day data for the metals, whose Yahoo close is the 13:30 ET COMEX settlement). '
        'Twelve specific data-leakage checks are enforced by a 34-case test suite.'))
    f.append(P(
        '<b>Result.</b> The validation-selected models are CatBoost for Bitcoin and Gold and a GRU for Silver. On the untouched test '
        'set all three assets sit at the random-walk floor: the served models\' return-RMSE is 0.0–0.6 % <i>above</i> the naive forecast '
        '(Diebold–Mariano p = 0.71–0.99), directional accuracy 46.9 % / 51.7 % / 55.2 % (binomial p = 0.83 / 0.37 / 0.12) and R² in '
        'return space ≈ 0. <b>No model beats the random walk at a one-day horizon for Bitcoin, Gold or Silver.</b> The result is '
        'consistent with weak-form market efficiency; the application reports the measured error next to every forecast. An '
        'intermediate version of the system reported 62 % directional accuracy for Silver; the final audit traced it to a close-time '
        'leak (same-day macro features against a 13:30 ET futures settlement), fixed it and re-ran the pipeline — the finding is '
        'reported in Section 9.'))
    f.append(P(
        '<b>Improvement over the initial version.</b> An audit of the original codebase found a price-reconstruction bug that '
        'corrupted every dollar metric, 30 % fabricated weekend rows for the metals, hyper-parameters selected on the test set, '
        'early stopping on training data, non-stationary price-level features, a served ensemble that predicted a constant, a '
        'broken inference path and misleading R² values (≈ 0.95 on price levels, which a random walk also achieves). All were fixed. '
        'A second, measured improvement phase then added 2.9× more training data (2018 →), volatility/momentum features, re-tuning, '
        'and logged design experiments; a final audit found and removed the close-time leak. On identical unseen days the served models '
        'moved by −0.6 % (Gold), −0.4 % (Silver) and +0.1 % (Bitcoin) in return-RMSE — inside noise: more data made the models equal to '
        'the random walk rather than slightly worse, not better than it.'))
    f.append(P(
        '<b>Status.</b> Working Streamlit dashboard (live forecast with uncertainty band, a <i>Predict-a-Day</i> demonstration on the '
        'unseen test period that reveals the actual price and the error, a full prediction history, validation/test tables, regime '
        'analysis and experiment tables) and FastAPI endpoints; reproducible pipeline (<font face="Mono">make pipeline</font>); '
        'documentation for methodology, features, results, limitations, demonstration and viva.'))
    return f


# ------------------------------------------------------------------ build
def build_report():
    doc = BaseDocTemplate(OUT_REPORT, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.2 * cm, bottomMargin=2.3 * cm,
                          title='Multi-Asset Next-Day Return Prediction — Final Year Project Report', author='Alyan Shahid (teamlocalhost)')
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='f')
    doc.addPageTemplates([PageTemplate(id='p', frames=[frame])])
    F = []

    # ---- cover
    F += [Spacer(1, 4.2 * cm),
          P('Multi-Asset Next-Day Return Prediction', 'title'),
          Spacer(1, 0.3 * cm),
          P('Bitcoin, Gold and Silver — a leakage-audited, walk-forward-validated comparison of statistical, '
            'tree-based and recurrent models against the random-walk baseline, with a live prediction dashboard and API', 'subtitle'),
          Spacer(1, 1.6 * cm),
          P('<b>Final Year Project Report</b>', 'cover'), Spacer(1, 0.2 * cm),
          P('Bachelor of Science in Computer Science', 'cover'), Spacer(1, 1.4 * cm),
          P('<b>Prepared by:</b> Alyan Shahid (team <i>teamlocalhost</i>)', 'cover'),
          P('<b>Supervisor:</b> ____________________', 'cover'),
          P('<b>Department of Computer Science, University of Lahore</b>', 'cover'),
          P('BSCS Fall 2022 – 2026', 'cover'), Spacer(1, 0.6 * cm),
          P(f'<b>Date:</b> {dt.date.today().strftime("%d %B %Y")}', 'cover'),
          Spacer(1, 2.4 * cm),
          P('Repository layout, methodology, feature dictionary, results, limitations and viva notes are also provided as '
            'Markdown under <font face="Mono">docs/</font>; every table in this report is generated from the files in '
            '<font face="Mono">results/</font> by <font face="Mono">docs/build_report.py</font>.', 'caption'),
          PageBreak()]

    # ---- 2 executive summary
    F += exec_summary_flowables()

    # ---- 3 introduction
    F.append(P('1. Introduction', 'h1'))
    F.append(P('Short-horizon financial forecasting is a standard test case for machine learning because it is genuinely hard: '
               'daily returns of liquid assets are dominated by noise, exhibit fat tails and volatility clustering, and any '
               'predictable component is small and unstable. Many student and published projects nevertheless report near-perfect '
               'accuracy — almost always because the evaluation is flawed (price-level R², random splits, test-set tuning, '
               'look-ahead in features). This project was built as a working prediction system for Bitcoin, Gold and Silver and '
               'then subjected to a full methodological audit. The final version answers a precise question — <i>does any of ten '
               'model families beat the random walk at a one-day horizon, and by how much?</i> — with a pipeline in which every '
               'leakage path is closed and tested, and it serves the validation-selected model in a dashboard that always shows '
               'the forecast next to its held-out performance.'))

    F.append(P('2. Project Objectives', 'h1'))
    F += bullets([
        'Build a reproducible, leakage-free pipeline from raw market data to next-day forecasts for three assets.',
        'Engineer stationary, financially motivated features from price, volume, macro and sentiment data.',
        'Compare baseline, classical, tree-based and deep models under expanding-window walk-forward validation, '
        'selecting hyper-parameters and the served model without touching the test set.',
        'Evaluate once on an untouched test period with metrics that are meaningful for returns (RMSE/MAE/R² in return space, '
        'directional accuracy with a significance test, Diebold–Mariano test against the random walk, strategy backtest).',
        'Deliver a demonstrable application (dashboard + API) that communicates the forecast, its horizon, the model used, '
        'its measured error and a clear disclaimer, and lets an evaluator predict any unseen day and compare with the actual value.',
        'Measure every design change (data size, features, horizon, target) on validation data before keeping it.',
        'Document the methodology, results, limitations and the audit-driven improvements for examination.'])

    F.append(P('3. Project Scope', 'h1'))
    F.append(P('In scope: daily data; a one-trading-day horizon; Bitcoin (BTC-USD), Gold (GC=F front-month futures) and Silver (SI=F); '
               'free public data sources; ten model families; a Streamlit dashboard and a FastAPI service running locally or in Docker. '
               'Out of scope: intraday data, order-book or news-text features, multi-day horizons, portfolio construction, and any form '
               'of trading execution. The system is a research prototype and decision-support tool, not financial advice.'))

    # ---- 4 system overview
    F.append(P('4. System Overview', 'h1'))
    F.append(P('The system is a single Python code base (Python 3.9, pandas, scikit-learn, LightGBM, CatBoost, TensorFlow/Keras 2.16, '
               'statsmodels, Streamlit 1.50, FastAPI) organised as a linear pipeline. Each stage is a script with a Make target; '
               '<font face="Mono">make pipeline</font> reproduces every number in this report from the raw CSV files.'))
    F.append(KeepTogether([workflow_drawing(), Paragraph('Figure 1. End-to-end workflow. The test split is read by exactly one script.', S['caption'])]))
    F.append(table([
        ['Stage', 'Module', 'What it does'],
        ['Data collection', 'src/data/data_collection.py, external_data.py', 'Daily OHLCV for the three assets; DXY, WTI, ^TNX, ^GSPC, ^VIX; Fear & Greed index'],
        ['Cleaning & features', 'src/data/preprocessing.py', 'De-duplication, trading-day calendar, log returns, indicators, ratio features, external alignment, frozen split, train-only scaler, build_dataset()'],
        ['EDA', 'src/data/eda.py', 'ADF tests on price vs return, ACF/PACF, return distributions, volatility clustering'],
        ['Models', 'src/models/registry.py, model_gru.py, model_lstm.py, ensemble_model.py', 'Uniform fit/predict interface for all models; two-phase fitting; stacked ensemble'],
        ['Tuning', 'src/training/tune_models.py', f'{CV_FOLDS}-fold expanding-window walk-forward grid search on train+val'],
        ['Training', 'src/training/train_models.py', 'Phase A (train → validation metrics, loss curves, importance) and phase B (refit on train+val)'],
        ['Evaluation', 'src/evaluation/backtesting.py, cross_validation.py, plots.py', 'Single test-set evaluation, model selection, results tables, figures'],
        ['Serving', 'src/inference/prediction.py, app/streamlit_app.py, src/api/app.py', 'Next-day forecast with context; dashboard; REST endpoints'],
        ['Experiments', 'src/experiments/run_experiments.py, before_after.py', 'E1 data size, E2 feature ablation, E3 horizon, E4 volatility (validation only); before/after on identical unseen days'],
        ['Quality', 'tests/ (34 cases), .github/workflows/ci.yml, Dockerfile', 'Look-ahead (crypto + futures), close-time alignment, split/embargo, reconstruction, metrics, demo-inference tests; CI; container'],
    ], [2.6 * cm, 5.0 * cm, 8.8 * cm]))
    F.append(Paragraph('Table 1. Implemented components (source of truth: the repository).', S['caption']))

    # ---- 5 dataset
    F.append(P('5. Dataset and Data Processing', 'h1'))
    F.append(table([
        ['Series', 'Source / ticker', 'Rows used', 'Period', 'Role'],
        ['Bitcoin OHLCV', 'Yahoo Finance, BTC-USD', '3,147 calendar days', '2018-01-31 → 2026-09-12', 'target + features'],
        ['Gold futures OHLCV', 'Yahoo Finance, GC=F', '2,156 exchange days', '2018-02-15 → 2026-09-11', 'target + features'],
        ['Silver futures OHLCV', 'Yahoo Finance, SI=F', '2,156 exchange days', '2018-02-15 → 2026-09-11', 'target + features'],
        ['US Dollar Index, WTI crude, 10-y yield', 'Yahoo Finance: DX-Y.NYB, CL=F, ^TNX', 'aligned to asset calendar', '2018-01 → 2026-09', 'metals features'],
        ['S&P 500, VIX', 'Yahoo Finance: ^GSPC, ^VIX', 'aligned', '2018-01 → 2026-09', 'all assets'],
        ['Crypto Fear & Greed index', 'alternative.me API', 'aligned', '2018-02 → 2026-09', 'Bitcoin feature'],
    ], [3.4 * cm, 4.3 * cm, 3.0 * cm, 3.4 * cm, 2.3 * cm]))
    F.append(Paragraph('Table 2. Data sources. Row counts are after indicator warm-up (30 rows). Raw downloads cover 2018-01-01 → 2026-09-12; '
                       'the earlier 3-year download (2023-07 → 2026-07) is archived for the before/after comparison.', S['caption']))
    F.append(P('<b>Cleaning</b> (<font face="Mono">DataCleaner.clean_data</font>): duplicate timestamps removed; rows sorted; non-positive '
               'prices dropped. Bitcoin is re-indexed to every calendar day (no gaps were present). Gold and Silver <b>keep their exchange '
               'calendar</b> — the original pipeline forward-filled weekends, which created ~30 % synthetic zero-return rows. External '
               'series are converted to daily log returns on their own calendar and aligned to the asset\'s calendar with a forward fill '
               '(only past values are carried forward). After feature computation the 30-row warm-up is dropped and the frame is validated '
               '(no NaN/inf).'))
    F.append(P('<b>Exploratory analysis</b> (training window only) confirms the modelling choices: price levels are non-stationary, returns '
               'are stationary, fat-tailed and almost uncorrelated (Table 3, Figure 3). This is why the target is the return and why the '
               'random walk is a demanding baseline.'))
    d = [['Asset', 'Days (total / train)', 'ADF p (price)', 'ADF p (return)', 'Daily σ %', 'Excess kurtosis', 'ACF lag-1']]
    for _, r in eda.iterrows():
        d.append([r['asset'], f"{int(r['n_days_total'])} / {int(r['n_days_train'])}", f"{r['adf_price_p']:.2f}", ('< 1e-10' if r['adf_return_p'] < 1e-10 else f"{r['adf_return_p']:.1e}"),
                  f"{r['ret_std_pct']:.2f}", f"{r['ret_excess_kurtosis']:.1f}", f"{r['acf_lag1']:+.3f}"])
    F.append(table(d, [2.0 * cm, 3.2 * cm, 2.4 * cm, 2.5 * cm, 2.0 * cm, 2.6 * cm, 2.0 * cm]))
    F.append(Paragraph('Table 3. Stationarity and distribution statistics of daily log returns (results/eda_summary.csv).', S['caption']))
    F.append(fig(os.path.join(FIGURES_DIR, 'price_history.png'), 15.5, 'Figure 2. Daily closes with the frozen train / validation / test windows.', max_height_cm=11.5))
    F.append(fig(os.path.join(FIGURES_DIR, 'eda_return_distributions.png'), 16.0, 'Figure 3. Return distributions are fat-tailed relative to a normal fit (training window).'))

    # ---- 6 features
    F.append(P('6. Feature Engineering', 'h1'))
    F.append(P('All features at day <i>t</i> use only information known at the moment the day-<i>t</i> close is fixed — the 00:00 UTC bar '
               'close for Bitcoin, the 13:30 ET COMEX settlement for Gold and Silver. Because the settlement precedes the S&amp;P 500 (16:00 ET), '
               'VIX (16:15), 10-year-yield (~15:00), DXY (17:00) and WTI (14:30) closes, and because Yahoo\'s futures High/Low span the whole '
               'Globex session, the metals use the <i>previous</i> day\'s macro returns and the previous session\'s High/Low-based features '
               '(hl_range, ATR/price, ADX); Bitcoin, whose bar closes after every US close, uses same-day values. Raw price levels (open, high, low, close, volume, EMA, '
               'Bollinger bands, ATR, MACD) are retained only for charts; the models receive scale-free ratios so that the feature '
               'distribution does not drift when the price trends outside the training range (Gold\'s test prices, $3,986–5,294, lie entirely '
               'above its training range of $1,817–3,644). A unit test perturbs the last observation and asserts that no earlier feature '
               'changes (no look-ahead).'))
    F.append(table([
        ['Group', 'Features (formula)', 'Assets'],
        ['Return path', 'log_return = ln(P_t/P_t−1); return_1d/2d/5d/10d = lagged returns; return_20d = 20-day cumulative return', 'all'],
        ['Volatility (rolling / HAR / EWMA)', 'volatility_10d/30d = rolling std; rv_1d/5d/22d = sqrt(mean r² over 1/5/22 days); ewma_vol = RiskMetrics σ (λ = 0.94)', 'all'],
        ['Momentum / trend', 'RSI(14); ADX(14); ROC(12); macd_norm = (EMA12−EMA26)/P; macd_hist_norm = (MACD−Signal9)/P; ema14_ratio, ema30_ratio = P/EMA−1', 'all'],
        ['Volatility state', 'bb_pctb = (P−BB_low)/(BB_up−BB_low); bb_width = (BB_up−BB_low)/BB_mid; atr_norm = ATR14/P; hl_range = (H−L)/P', 'all'],
        ['Macro', 'sp500_return, vix_return (all); dxy_return, oil_return, tnx_return (metals); gold_return (silver). Previous-day values for the metals (close-time rule)', 'per asset'],
        ['Sentiment / calendar / activity', 'fear_greed (0–100); dow_sin, dow_cos; log_volume_change (clipped ±3)', 'Bitcoin'],
    ], [3.2 * cm, 10.9 * cm, 2.3 * cm]))
    F.append(Paragraph('Table 4. Model input features: Bitcoin 29, Gold 28, Silver 29 (docs/FEATURES.md gives every formula and rationale). Groups added in the improvement phase are in the second row.', S['caption']))
    F.append(P('Deliberately excluded: futures volume for Gold/Silver (front-month contract volume from Yahoo jumps at contract rolls), '
               'day-of-week for the metals (no justification once synthetic weekends are removed), and redundant level indicators (VWAP, SMA, BB_Mid).'))
    F.append(P('<b>Target.</b> <i>y<sub>t</sub> = ln(P<sub>t+1</sub>/P<sub>t</sub>)</i>, the next row\'s log return, created only inside '
               '<font face="Mono">create_sequences</font>; the 30-day input window never contains that row (unit-tested). Recurrent models '
               f'receive the {SEQ_LEN}×F window; tabular models receive the last row of the same window, so every model is scored on '
               'identical days. The target is standardised with the training mean and standard deviation and inverted exactly for reporting.'))

    # ---- 7 models
    F.append(P('7. Machine Learning Models', 'h1'))
    F.append(table([
        ['Family', 'Model', 'Input', 'Notes'],
        ['Baseline', 'Naive (random walk)', '—', 'ŷ = 0, i.e. tomorrow\'s price = today\'s. The reference every model must beat.'],
        ['Baseline', 'Historical mean', '—', 'ŷ = mean training return (drift).'],
        ['Baseline', 'ARIMA(p,0,q)', 'return series', 'Order by AIC on train (BTC (1,0,0), Gold (2,0,2), Silver (3,0,2)); one-step walk-forward.'],
        ['Linear', 'Ridge', 'last-day features', 'L2-regularised regression; α tuned.'],
        ['Trees', 'Random Forest', 'last-day features', 'Depth-limited bagged trees.'],
        ['Trees', 'LightGBM', 'last-day features', 'Gradient boosting, early-stopped on validation.'],
        ['Trees', 'CatBoost', 'last-day features', 'Ordered boosting, early-stopped on validation. Served for Bitcoin and Gold.'],
        ['Deep', 'GRU', '30×F window', 'One GRU layer (32 units) + dropout + dense; early stopping. Served for Silver.'],
        ['Deep', 'LSTM', '30×F window', 'Same shape as the GRU for a fair comparison.'],
        ['Ensemble', 'Stacked', 'OOF predictions', 'Non-negative Ridge over Ridge/LightGBM/CatBoost/GRU (experiment).'],
    ], [1.8 * cm, 3.1 * cm, 2.6 * cm, 8.9 * cm]))
    F.append(Paragraph('Table 5. Models compared (src/models/registry.py).', S['caption']))
    F.append(P('<b>Two-phase training.</b> Every model with a stopping rule (number of trees / boosting iterations / epochs) is first fitted on '
               'the training window while monitoring validation loss to find the stopping point (phase A — this also yields honest '
               'validation metrics and loss curves), then refitted on train+validation with that stopping point fixed (phase B — the deployed '
               'model). The validation data is therefore never inside its own early-stopping monitor.'))
    hp = [['Asset', 'Served model & tuned parameters', 'Stopping point', 'Other tuned configurations']]
    bp = best['return_1d']
    for a in ASSETS:
        m = status[a]['primary_model']; p = bp[a][m]; fi = status[a]['fit_info']
        desc = {'CatBoost': lambda: f"CatBoost: depth {p['depth']}, lr {p['learning_rate']}, L2 {p['l2_leaf_reg']}",
                'LightGBM': lambda: f"LightGBM: {p['num_leaves']} leaves, lr {p['learning_rate']}, min-leaf {p['min_child_samples']}, λ {p['reg_lambda']}",
                'GRU': lambda: f"GRU: {p['units']} units, dropout {p['dropout']}, lr {p['learning_rate']}, batch {p['batch_size']}",
                'LSTM': lambda: f"LSTM: {p['units']} units, dropout {p['dropout']}, lr {p['learning_rate']}, batch {p['batch_size']}",
                'Ridge': lambda: f"Ridge: α {p['alpha']}",
                'RandomForest': lambda: f"Random Forest: depth {p['max_depth']}, min-leaf {p['min_samples_leaf']}"}[m]()
        stop = (f"{fi['epochs_used']} epochs" if 'epochs_used' in fi else
                f"{fi.get('iterations_used', fi.get('n_estimators_used', '—'))} rounds" if fi else '—')
        others = [f"Ridge α {bp[a]['Ridge']['alpha']}", f"RF depth {bp[a]['RandomForest']['max_depth']}",
                  f"LightGBM {bp[a]['LightGBM']['num_leaves']} leaves", f"CatBoost depth {bp[a]['CatBoost']['depth']}",
                  f"GRU {bp[a]['GRU']['units']}u/drop {bp[a]['GRU']['dropout']}", f"LSTM {bp[a]['LSTM']['units']}u/drop {bp[a]['LSTM']['dropout']}"]
        hp.append([a, desc, stop, '; '.join(o for o in others if not o.startswith({'CatBoost': 'CatBoost', 'LightGBM': 'LightGBM', 'GRU': 'GRU', 'LSTM': 'LSTM', 'Ridge': 'Ridge', 'RandomForest': 'RF'}[m]))])
    F.append(table(hp, [1.7 * cm, 5.3 * cm, 2.2 * cm, 7.2 * cm]))
    F.append(Paragraph('Table 6. Hyper-parameters chosen by walk-forward validation (results/tuning/best_params.json). All tuned configurations '
                       'of a model lie within 0.4–2 % of each other while the fold-to-fold standard deviation is ≈ 20 % of the mean; early '
                       'stopping selected 10–82 boosting rounds for the metals\' LightGBM/CatBoost (Gold CatBoost 606 at learning-rate 0.01) and '
                       '1–9 epochs for the recurrent models: the data supports very little capacity.', S['caption']))
    F.append(P('<b>Which model is served, and why.</b> Selection is automatic: the model with the lowest mean walk-forward RMSE of '
               'the return on the train+val folds — CatBoost for Bitcoin and Gold, a GRU for Silver. All nine models lie within 0.4–0.5 % of '
               'each other and within 0.35 % of the random walk on validation, so the ranking is not statistically decisive and is stated as '
               'such. The served Silver GRU illustrates the point: its predictions have a standard deviation of 0.0009 against a realised 0.032 — '
               'it is effectively a drift forecast that scored lowest on the folds. Boosted trees remain the practical choice (seconds to train, '
               'feature importance); the recurrent models stop after 1–9 epochs because validation loss never improves.'))

    # ---- 8 training & evaluation
    F.append(P('8. Training and Evaluation Methodology', 'h1'))
    F.append(table([
        ['Split', 'Rule', 'Bitcoin', 'Gold / Silver', 'Used for'],
        ['Train', f'target ≤ {TRAIN_END}', '2,750 windows (from 2018-03)', '1,874 windows (from 2018-03)', 'fitting; scaler'],
        ['Validation', f'{TRAIN_END} < target ≤ {VAL_END}', '160 days', '109 days', 'early stopping; walk-forward folds with train'],
        ['Test', f'target > {VAL_END} (to 2026-09-12)', '207 days', '143 days', 'evaluated once by backtesting.py'],
    ], [2.0 * cm, 4.2 * cm, 3.4 * cm, 3.4 * cm, 3.4 * cm]))
    F.append(Paragraph('Table 7. Frozen chronological split (config.py), defined by the dates the target covers (exact embargo for multi-day targets). '
                       'No shuffling; identical calendar boundaries for all assets.', S['caption']))
    F.append(P(f'<b>Walk-forward validation.</b> The train+validation period is cut into {CV_FOLDS} consecutive validation blocks; for fold '
               '<i>k</i> the model is trained on everything before the block and scored on the block (expanding window). Inside each fold '
               'the last 15 % of the fold\'s training window is the early-stopping monitor. Hyper-parameter grids (over regularisation '
               'strength rather than capacity) and model selection use only these folds. k-fold cross-validation was rejected because it '
               'would train on 2026 to predict 2024 and leak regime information.'))
    F.append(table([
        ['Metric', 'Space', 'Why it is reported'],
        ['MAE, RMSE, MAPE', 'USD price', 'What a user experiences; MAPE is scale-free across a $60,000 and a $60 asset.'],
        ['RMSE, MAE, R²', 'log return', 'The quantity actually predicted; R² &lt; 0 means worse than the mean. R² on price levels is NOT reported because a random walk scores ≈ 0.95 there.'],
        ['Directional accuracy', 'sign', '% of non-flat days with sign(ŷ) = sign(y), with a one-sided binomial p-value against 50 %. A zero forecast has no direction ("—").'],
        ['Diebold–Mariano test', 'log return', 'Formal test of equal squared-error accuracy against the random walk (HAC variance, small-sample correction).'],
        ['Strategy backtest', '—', 'Long/flat rule (long if ŷ &gt; 0) with 10 bps cost vs buy-and-hold — economic relevance.'],
        ['Train vs validation RMSE', 'log return', 'Over-fitting gap, compared with the random walk\'s own gap.'],
    ], [3.4 * cm, 2.0 * cm, 11.0 * cm]))
    F.append(Paragraph('Table 8. Evaluation metrics (src/utils/metrics.py).', S['caption']))
    F.append(table([
        ['Leakage check', 'Mechanism', 'Verified by'],
        ['Random shuffling', 'chronological date slicing', 'test_split_is_chronological_and_disjoint'],
        ['Scaler fitted on val/test', 'MinMaxScaler.fit(train) only', 'test_scaler_fitted_on_train_only'],
        ['Target inside input window', 'y = row i+30; window = rows i…i+29', 'test_create_sequences_target_is_next_row_and_never_in_window'],
        ['Look-ahead in rolling / EWM / shift features', 'all operators backward-looking', 'test_no_lookahead_in_any_feature (crypto and futures paths)'],
        ['External data closing after the asset (close-time rule)', 'same-day for BTC (00:00 UTC bar); latest value strictly before day t for the metals (13:30 ET settlement)', 'test_commodity_external_features_are_previous_day_values'],
        ['Futures High/Low past the settlement', 'hl_range, atr_norm, ADX lagged one session for the metals', 'test_commodity_high_low_features_are_lagged_one_session'],
        ['Early stopping on training data', 'two-phase fit', 'registry.py; CatBoost bug removed'],
        ['Hyper-parameters chosen on test', 'tuner reads train+val arrays only', 'grep: no test reference in training/tuning/stacking'],
        ['Synthetic rows', 'exchange calendar for futures', 'test_commodities_keep_trading_calendar'],
        ['Wrong price reconstruction', 'explicit previous close', 'test_true_target_reconstructs_actual_close_exactly (error 1e-11)'],
        ['Test set touched more than once', 'single script', 'backtesting.py is the only reader of the test split'],
        ['Multi-day target straddling a split', 'split by target dates', 'test_task_targets_are_future_only_and_embargoed'],
        ['Demo prediction seeing future rows', 'features.loc[:as_of] before scaling', 'test_predict_for_date_… perturbs every future row; forecast unchanged'],
    ], [4.2 * cm, 5.0 * cm, 7.2 * cm]))
    F.append(Paragraph('Table 9. Data-leakage checklist enforced by the test suite (tests/, 34 cases, all passing).', S['caption']))

    # ---- 9 results
    F.append(P('9. Experimental Results', 'h1'))
    F.append(P('Test period 2026-02-18 → 2026-09-12 for all assets (Bitcoin 207 days; Gold and Silver 143 exchange days). '
               '"CV" columns are the walk-forward validation scores used for selection; the remaining columns are the single test-set '
               'evaluation. The served model is marked "(served)". All values are from results/cv_results.csv and results/final_test_results.csv.'))
    for i, a in enumerate(ASSETS):
        F.append(KeepTogether([P(f'9.{i + 1} {a}', 'h2'), results_table(a),
                               Paragraph(f'Table {10 + i}. {a}: walk-forward validation and untouched test-set results, all models.', S['caption'])]))
    F.append(fig(os.path.join(FIGURES_DIR, 'model_comparison_test.png'), 16.4,
                 'Figure 4. Test-set comparison: RMSE of the next-day return (red line = random walk) and directional accuracy (red line = 50 %).'))
    F.append(P('9.4 Interpretation', 'h2'))
    F += bullets([
        '<b>All three assets are at the random-walk floor.</b> The served models\' return-RMSE is 0.0–0.6 % above the naive forecast '
        '(Diebold–Mariano p = 0.74 Bitcoin, 0.71 Gold, 0.99 Silver); directional accuracy 46.9 % / 51.7 % / 55.2 % (binomial p = 0.83 / 0.37 / 0.12); '
        'R² in return space −0.004 / −0.015 / −0.001. The tuned models are close to the unconditional drift, and the USD errors '
        '(MAPE 1.53 / 1.30 / 2.48 %) equal the random walk\'s (1.52 / 1.29 / 2.49 %) because they measure the asset\'s volatility, not skill.',
        '<b>The one p &lt; 0.05 cell.</b> Silver LightGBM scores 58.7 % directional accuracy (p = 0.02) but has exactly the random walk\'s RMSE '
        '(DM p = 0.98) and a walk-forward figure of 53.3 %; with 27 model-rows in the test table one such cell is what chance produces '
        '(Bonferroni threshold ≈ 0.002). It is not the validation winner and is not claimed.',
        '<b>The leak that was found and removed (Section 9.6).</b> An intermediate version of this system aligned the metals\' macro features '
        'same-day and reported 62.2 % directional accuracy for Silver (DM p = 0.002); Yahoo\'s futures close is the 13:30 ET settlement, so those '
        'features contained 1–3.5 hours of the target interval. Removing them dropped the figure to 51.7 %; the corrected alignment leaves the '
        'served model at 55.2 %. Bitcoin, whose bar closes after the US close, was never affected — its numbers are identical before and after.',
        '<b>Deep models.</b> GRU and LSTM are the weakest family on every asset in validation and not better than the random walk in test '
        '(early stopping after 1–12 epochs), even with 3× the earlier data.',
        '<b>Stacked ensemble.</b> With 3 years of data the non-negative meta-weights collapsed to zero; with the full history they are '
        f'non-zero (Bitcoin: LightGBM {stack["Bitcoin"]["weights"]["LightGBM"]:.2f}, CatBoost {stack["Bitcoin"]["weights"]["CatBoost"]:.2f}; '
        f'Gold: CatBoost {stack["Gold"]["weights"]["CatBoost"]:.2f}, GRU {stack["Gold"]["weights"]["GRU"]:.2f}; '
        f'Silver: GRU {stack["Silver"]["weights"]["GRU"]:.2f}, LightGBM {stack["Silver"]["weights"]["LightGBM"]:.2f}) '
        'but the stack does not beat the best single model on test (for Gold it is the worst model there): the base models carry no '
        'combinable out-of-sample signal.',
        '<b>Over-fitting.</b> Validation RMSE is 0.8× (Bitcoin), 2.1× (Gold) and 2.9× (Silver) the training RMSE — and the random walk shows '
        'the same ratios (Figure 6): the gap is regime shift, not memorisation. The one model that visibly over-fits is Bitcoin\'s LightGBM '
        '(training RMSE 0.0274 against 0.0335 for the random walk, with no validation gain).'])

    # ---- regime table
    F.append(P('9.5 Performance across market regimes (test period, served model vs naive)', 'h2'))
    rg = [['Asset', 'Regime', 'Days', 'MAE % served', 'MAE % naive', 'Direction hits %']]
    for a in ASSETS:
        for _, r in REG[(REG.asset == a) & (REG.regime_type != 'day_regime')].iterrows():
            rg.append([a, r['regime'], int(r['n_days']), f"{r['mae_pct_served']:.2f}", f"{r['mae_pct_naive']:.2f}", f"{r['dir_hit_served_pct']:.1f}"])
    t = table(rg, [1.8 * cm, 4.2 * cm, 1.4 * cm, 2.8 * cm, 2.8 * cm, 3.4 * cm], font=7.8)
    t.setStyle(TableStyle([('ALIGN', (2, 1), (-1, -1), 'RIGHT')]))
    F.append(t)
    F.append(Paragraph('Table 13. Regimes are defined on the day the prediction is made (30-day volatility tercile; sign of the 20-day return). '
                       'Errors scale with volatility for every asset; no served model improves on the naive forecast in any slice (results/regime_analysis.csv).', S['caption']))

    # ---- before / after
    F.append(P('9.6 Before vs after the improvement phase (identical unseen days 2026-02-18 → 2026-07-28)', 'h2'))
    F.append(P('The archived 3-year system and the improved system are compared on exactly the same days. Yahoo Finance revised a few '
               'historical closes between the two downloads (mean &lt; 0.005 %, max 0.75 % on one Bitcoin day), so each system is also shown '
               'against the naive forecast computed on its own data (results/experiments/before_after.csv).'))
    ba = [['Asset', 'System', 'Model', 'MAE ($)', 'RMSE ($)', 'MAPE %', 'RMSE (ret)', 'R² (ret)', 'Dir. acc. % (p)', 'DM p']]
    for a in ASSETS:
        for lab in ('before (3 y data)', 'after (full history)', 'naive (new data)'):
            r = BA[(BA.asset == a) & (BA.system == lab)].iloc[0]
            da = '—' if r['model'] == 'Naive' else f"{r['DirAcc_pct']:.1f} ({r['DirAcc_pvalue']:.2f})"
            dm = '—' if r['model'] == 'Naive' else f"{r['DM_pvalue_vs_naive']:.2f}"
            ba.append([a, lab, r['model'], f"{r['MAE_usd']:,.2f}", f"{r['RMSE_usd']:,.2f}", f"{r['MAPE_usd']:.2f}", f"{r['RMSE_ret']:.5f}",
                       f"{r['R2_ret']:+.3f}", da, dm])
    t = table(ba, [1.6 * cm, 3.0 * cm, 1.9 * cm, 1.6 * cm, 1.7 * cm, 1.3 * cm, 1.7 * cm, 1.4 * cm, 1.9 * cm, 1.1 * cm], font=7.4)
    t.setStyle(TableStyle([('ALIGN', (3, 1), (-1, -1), 'RIGHT')]))
    F.append(t)
    F.append(Paragraph('Table 14. Gold −0.6 %, Silver −0.4 %, Bitcoin +0.1 % return-RMSE — all inside noise. The served models now sit on the '
                       'random-walk line instead of 0.6–1.5 % below it; the intermediate claim of a 2.8 % Silver improvement was the close-time leak. '
                       'This comparison reads the test rows a second time; it compares two frozen systems and selects nothing.', S['caption']))

    # ---- the leak found by the final audit
    F.append(P('9.6b The close-time leak found by the final audit', 'h2'))
    F.append(P('Yahoo\'s daily close for GC=F and SI=F is the 13:25–13:30 ET COMEX <i>settlement</i>, not the 17:00 ET end of the Globex session. '
               'This was verified against 5-minute bars of the December-2026 contracts: the mean absolute difference between the daily close and the '
               '13:30 price is 0.04 % (gold) / 0.09 % (silver), against 0.3–1 % for the 17:00 price. The intermediate version of this system used '
               'the <i>same-day</i> S&amp;P 500, VIX, 10-year-yield, DXY and WTI returns — fixed between 14:30 and 17:00 ET — and the session '
               'High/Low as day-<i>t</i> features for the metals. Those values contain up to 3.5 hours of the interval the target measures.'))
    F.append(table([
        ['Silver, same 143 unseen days', 'Model', 'Dir. acc. (p)', 'DM p vs naive', 'R² (return)'],
        ['same-day macro features (archived, leak)', 'LightGBM', '62.2 % (0.002)', '0.002', '+0.044'],
        ['the five macro features removed', 'LightGBM', '51.7 % (0.37)', '0.21', '+0.015'],
        ['previous-day macro + lagged High/Low (final)', 'GRU (served)', '55.2 % (0.12)', '0.99', '−0.001'],
    ], [6.4 * cm, 2.6 * cm, 2.6 * cm, 2.4 * cm, 2.4 * cm]))
    F.append(Paragraph('Table 14b. The Silver "edge" was the leak (results/archive_sameday_macro_leak/). The effect was invisible in walk-forward '
                       'validation (dropping the macro group changed CV RMSE by −0.04 %) and only surfaced in a test window where the metals\' '
                       'contemporaneous correlation with equities and rates doubled — a leak that is small on average can dominate one volatile window.', S['caption']))
    F.append(P('The fix (config.EXTERNAL_SAME_DAY, DataCleaner._align_external, lag_post_settlement_features) uses, for commodity assets, the latest '
               'external value <i>strictly before</i> day <i>t</i> and the previous session\'s High/Low-based features; Bitcoin keeps same-day values '
               'because its bar closes at 00:00 UTC, after every US close (and its results are bit-identical before and after the fix). Two unit '
               'tests enforce the rule. The lesson is stated as a finding of the project: "backward-looking by date" is not the same as "known when '
               'the target starts", and a leak of a few hours is enough to manufacture a significant result on daily data.'))

    # ---- design experiments
    F.append(P('9.7 Design experiments (validation only)', 'h2'))
    e1 = EXP['E1']; e3 = EXP['E3']; e4 = EXP['E4']; e2 = EXP['E2']
    def e1row(a, m):
        x = e1[(e1.asset == a) & (e1.model == m)]; return x.iloc[0]['val_RMSE_ret'], x.iloc[1]['val_RMSE_ret']
    F += bullets([
        '<b>E1 data size</b> (same features and parameters, fixed validation window 2025-09 → 2026-02): training on 2018 → instead of 2023 → '
        + '; '.join(f"{a} CatBoost {(e1row(a, 'CatBoost')[1] / e1row(a, 'CatBoost')[0] - 1) * 100:+.1f} %" for a in ASSETS)
        + ' validation RMSE — Bitcoin benefits, the metals are neutral; kept because it also stabilises tuning.',
        '<b>E2 feature-group ablation</b> (walk-forward CV, Ridge and LightGBM): dropping any one of the five groups changes RMSE by '
        f"{e2[e2.dropped_group != '(none — all features)']['delta_RMSE_pct'].min():+.2f} % … {e2[e2.dropped_group != '(none — all features)']['delta_RMSE_pct'].max():+.2f} % — inside noise; "
        'the full set is kept for interpretability.',
        '<b>E3 horizon</b>: a 5-day return target is not more predictable than the next day (best model '
        f"{e3[(e3.task == 'return_5d') & (e3.model != 'Naive')]['RMSE_vs_naive_pct'].min():+.2f} % vs naive); the served horizon stays 1 day.",
        '<b>E4 volatility</b>: 22-day realised volatility is more predictable than the return (Bitcoin Ridge '
        f"{e4[(e4.asset == 'Bitcoin') & (e4.model == 'Ridge')]['RMSE_vs_persistence_pct'].iloc[0]:+.1f} % vs persistence) but only marginally better than a RiskMetrics EWMA "
        f"({e4[(e4.asset == 'Bitcoin') & (e4.model == 'EWMA')]['RMSE_vs_persistence_pct'].iloc[0]:+.1f} %) and worse than EWMA for Silver — documented, not productised."])
    F.append(fig(os.path.join(FIGURES_DIR, 'silver_actual_vs_predicted.png'), 15.2,
                 'Figure 5. Silver test period, served GRU: predicted vs actual next-day returns (top, scatter middle) and the price view (bottom). '
                 'A price line always tracks the actual with a one-day lag; skill — or, here, its absence — is only visible in return space.', max_height_cm=12.5))
    F.append(fig(os.path.join(FIGURES_DIR, 'overfitting_gap.png'), 15.0,
                 'Figure 6. Train vs validation RMSE per model with the random walk\'s train/validation levels (dotted/dashed).'))
    F.append(fig(os.path.join(FIGURES_DIR, 'gold_feature_importance.png'), 10.5,
                 'Figure 7. Gold, served CatBoost: normalised feature importance (the served Silver GRU exposes none).'))

    # ---- 10 application
    F.append(P('10. Final System and Application', 'h1'))
    F.append(P('The Streamlit dashboard (<font face="Mono">make serve</font>, port 8501) and the FastAPI service '
               '(<font face="Mono">make api</font>, port 8000) sit on the same inference module. Inference scales the features up to the '
               'chosen day with the train-fitted scaler, runs the requested model on the last 30 rows, un-standardises the predicted return and '
               'converts it to a price. The <i>Predict a Day</i> tab is the demonstration: stand on any day of the unseen test period, generate '
               'the next-day forecast from data up to that day only, then reveal the actual close, the error in dollars and percent, and the '
               'direction hit — highlighted on the actual-vs-predicted chart, with every other model\'s forecast for the same day. The '
               '<i>Prediction History</i> tab lists every unseen day with predicted, actual, error and hit/miss, plus MAE and hit-rate KPIs. '
               'A "Sync Live Market Data" button refreshes features into a separate live file so the frozen evaluation data is never overwritten.'))
    F.append(table([
        ['Implemented (verified running)', 'Not implemented (future possibilities)'],
        ['• Select Bitcoin / Gold / Silver; price chart with train/val/test shading; Bollinger, RSI, MACD, volume overlays<br/>'
         '• <b>Predict a Day (unseen test)</b>: choose any test day → predicted close, actual close, error $ and %, direction hit/miss, chart highlight, all models on that day<br/>'
         '• <b>Prediction History</b>: every unseen day with predicted / actual / error / hit, MAE and hit-rate KPIs, CSV download<br/>'
         '• Run a next-day prediction with the served model or any trained model: current close, predicted close, % change, direction, '
         'horizon (T+1), model name, ±1 RMSE uncertainty band, held-out metrics of that model, disclaimer<br/>'
         '• Forecast-horizon chart (last 60 days + next day); table of all trained models\' predictions on the same input<br/>'
         '• Validation view: actual vs predicted returns and prices over the test period; return scatter<br/>'
         '• Performance tab: walk-forward and test tables, honest-reading summary, regime table, experiment tables (E1–E4), report figures<br/>'
         '• Methodology tab: task, split, features, models, served-model parameters, model inventory<br/>'
         '• API: GET /health, GET /models, GET /predict/{asset}[?model=…]; Docker Compose for both services<br/>'
         '• Deep links: ?asset=Gold&amp;run=1',
         '• Multi-day horizons or volatility forecasts<br/>• Calibrated prediction intervals (the band is an ex-post error band)<br/>'
         '• User accounts, saved forecasts, database<br/>• Automatic scheduled retraining in production<br/>'
         '• News / NLP sentiment for the metals<br/>• Intraday data<br/>• Trading execution or portfolio management'],
    ], [10.8 * cm, 5.6 * cm]))
    F.append(Paragraph('Table 15. Application functionality.', S['caption']))
    demo_crop = crop_image(os.path.join(DOC_FIG, 'dash_demo_silver.png'), os.path.join(DOC_FIG, 'dash_demo_silver_crop.png'), (0, 0, 2100, 2150))
    F.append(fig(demo_crop, 15.0, 'Figure 8. Dashboard, Silver — Predict a Day: the served GRU stands on 2026-09-10, predicts $64.34 for 2026-09-11 '
                 'from data up to that day, the actual close ($64.55) is revealed, error −0.33 %, direction hit; all seven models on the same day '
                 'below, within $0.08 of each other (screenshot of the running application).', max_height_cm=16.0))
    hist_crop = crop_image(os.path.join(DOC_FIG, 'dash_history_silver.png'), os.path.join(DOC_FIG, 'dash_history_silver_crop.png'), (0, 0, 2100, 1500))
    F.append(fig(hist_crop, 15.0, 'Figure 9. Dashboard, Silver — Prediction History: every unseen day with predicted, actual, error and hit/miss; MAE and hit-rate KPIs.', max_height_cm=11.5))
    perf_crop = crop_image(os.path.join(DOC_FIG, 'dash_performance.png'), os.path.join(DOC_FIG, 'dash_performance_crop.png'), (0, 0, 2100, 1700))
    F.append(fig(perf_crop, 15.0, 'Figure 10. Dashboard — Model Performance tab with the walk-forward and test tables generated by the evaluation script.', max_height_cm=12.5))

    # ---- 11 improvements
    F.append(P('11. Improvements Made During the Audit', 'h1'))
    F.append(P('The initial project already generated predictions, used a log-return target, a chronological split and a train-only scaler, '
               'and had a modular code base with a dashboard. The audit nevertheless found problems that invalidated its reported results. '
               'All were fixed; the original artefacts are archived under results/archive_pre_fix/ for the discussion.'))
    F.append(table([
        ['Area', 'Original problem', 'Final implementation'],
        ['Price reconstruction', 'Predicted price rebuilt from the previous day\'s open (column 0) instead of the close — every USD metric and every dashboard forecast wrong (≈ $1,416 RMSE for Bitcoin from the bug alone)', 'Explicit previous close; regression test asserts exact reconstruction of the true target'],
        ['Data pipeline', 'Gold/Silver forward-filled to calendar days: 30.7 % fabricated zero-return rows, 47 of 159 test days fake', 'Exchange calendar kept for futures; macro data aligned onto it'],
        ['Data leakage', 'Hyper-parameters chosen on the test set (April experiments); CatBoost early-stopped on data inside its training set', 'Walk-forward tuner on train+val only; two-phase fitting; checklist enforced by tests'],
        ['Features', 'Raw price levels (open/high/low/close/EMA/BB/VWAP) as inputs — test values 1.7–2.4× outside the training range', 'Stationary ratio features; redundant indicators removed; gold return added for silver'],
        ['Model', 'Served "stacked ensemble" had coefficients ≈ 0 (Bitcoin exactly [0,0,0]) — a constant predictor, never evaluated on test; 2×100-unit GRU/LSTM over-fitted', 'Uniform registry of 10 models; validation-based selection; small tunable GRU/LSTM; stack kept as a reported experiment'],
        ['Evaluation', 'R² ≈ 0.94–0.97 on price levels presented as skill; naive baseline beat every model but was not acknowledged; directional accuracy mis-computed (naive 0 %)', 'Return-space metrics; correct directional accuracy with p-values; Diebold–Mariano test; strategy backtest; over-fitting gap; honest reporting'],
        ['Serving', 'CLI/API crashed (LightGBM given 900 features; float(dict)); dashboard listed stale LSTM and a non-existent Ensemble path', 'Rewritten inference, API and dashboard around the served model with context and disclaimer; verified in the browser'],
        ['Code & docs', 'Five stale result files from an earlier pipeline; Makefile/CI referenced missing scripts; 4 of 31 tests failed; README claimed models/indicators that did not exist', 'Single results/ tree; working Makefile/CI/retrain; 31 passing tests; accurate README and docs/; bit-identical reproducibility'],
        ['Improvement phase', '3 years of data; no measured design decisions; forecast shown without a way to verify it', '2018 → data (2.9×); HAR/EWMA volatility and momentum features; per-task tuning; logged experiments E1–E4; before/after on identical days; Predict-a-Day demo and prediction history'],
        ['Final audit', 'Same-day macro and session High/Low features for the metals against a 13:30 ET settlement close → a spurious 62 % Silver result; live sync overwrote frozen raw files; make targets missing a path bootstrap', 'Close-time rule (previous-day macro, lagged High/Low for metals) with unit tests; pipeline re-run; contaminated results archived; live downloads isolated in data/raw_live/; reproducible make targets'],
    ], [2.6 * cm, 7.4 * cm, 6.4 * cm], font=7.8))
    F.append(Paragraph('Table 16. Before vs final implementation (docs/FIX_PLAN.md, docs/AUDIT_SUMMARY.md, docs/RESULTS.md §2–3).', S['caption']))

    # ---- 12 limitations
    F.append(P('12. Limitations', 'h1'))
    F += bullets([
        '<b>Market efficiency dominates.</b> Lag-1 autocorrelation of returns is −0.08 / −0.04 / +0.00 (BTC / Gold / Silver); at a one-day '
        'horizon the predictable component is small and unstable, and the results are consistent with that.',
        '<b>One test window, many comparisons.</b> 8.6 years of training data removed the worst of the small-sample problem, but the unseen window is still a '
        'single 5–7-month period (143–207 days): a few percent in RMSE and directional accuracies of 52–57 % are not statistically detectable, '
        'and with 27 model-rows in the test table one cell at p ≈ 0.02 is expected by chance.',
        '<b>Regime shift in the test window.</b> The 2026 window is far more volatile than the 2018–2025 average (validation RMSE 2–3× training for '
        'the metals, for the random walk as well as for the models); one window is a single draw from a non-stationary process.',
        '<b>Data quality.</b> Front-month futures (roll effects; unusable volume), a composite BTC index, forward-filled macro series on holidays, '
        'a proprietary sentiment index, and vendor revisions of historical closes between downloads (max 0.75 %).',
        '<b>Modelling.</b> One-day point forecasts only; the dashboard\'s band is an ex-post RMSE band, not a calibrated interval; small tuning '
        'grids; selection by RMSE (directional accuracy and Sharpe are reported but not optimised).',
        '<b>Backtest.</b> The long/flat strategy uses a flat 10 bps cost with no slippage, shorting, sizing or financing.'])

    F.append(P('13. Future Work', 'h1'))
    F += bullets([
        'Roll the test window forward as new months arrive and re-evaluate walk-forward; use the exchange\'s official settlement series rather than a free vendor\'s bar.',
        'Forecast volatility (GARCH-type or quantile targets), where the volatility-regime features clearly carry information, and score calibration.',
        'Multi-day horizons and probabilistic forecasts; regime-aware or switching models.',
        'News/NLP sentiment for the metals; intraday data; a realistic transaction-cost model.'])

    F.append(KeepTogether([P('14. Conclusion', 'h1'), P('The project set out to build a working next-day prediction system for Bitcoin, Gold and Silver and to make it '
               'scientifically defensible. The final system has a leakage-audited, reproducible pipeline; ten model families compared '
               'under expanding-window walk-forward validation on identical days; a single, statistically tested evaluation on an untouched '
               'period; an evaluator-facing demonstration that predicts any unseen day and reveals the actual value and the error; and a '
               'dashboard and API that serve the validation-selected model together with its measured error. The empirical answer with 8.6 '
               'years of data is unambiguous and honest: for Bitcoin, Gold and Silver no statistical, tree-based or recurrent model beats the '
               'random walk at a one-day horizon (served models 0.0–0.6 % above the naive RMSE, Diebold–Mariano p ≥ 0.71, directional accuracy '
               '47–55 %). The 2.9× larger training set moved the models from slightly below the random walk onto it, not above it. The value of '
               'the work lies in the rigour of the evaluation — in showing why the original "R² ≈ 0.95" was an illusion, and in finding and '
               'removing a close-time leak in its own intermediate version that had manufactured a "significant" 62 % result — rather than in a '
               'claimed forecasting edge.')]))

    F.append(P('15. FYP Readiness Assessment', 'h1'))
    F.append(P('<b>Suitable for submission and live demonstration.</b> The implementation demonstrates the complete ML research workflow '
               '(problem → data → analysis → features → baselines → models → validation → experiments → comparison → final model → '
               'system → evaluation → limitations); every methodological claim is backed by a test, a script or a results file; the '
               'application runs for all three assets with an interactive unseen-day demonstration; every design change was measured before '
               'being kept; and the pipeline reproduces every reported number. The remaining weakness is the single test window, which is '
               'stated in the limitations. It should be presented as a methodological benchmark and decision-support prototype — including the '
               'leak it found in itself — not as a market-beating predictor.', 'box'))

    F.append(P('Appendix — Reproduction', 'h1'))
    F.append(P('python -m venv venv &amp;&amp; source venv/bin/activate &amp;&amp; pip install -r requirements.txt<br/>'
               'make preprocess eda tune train stack evaluate figures experiments test   # full pipeline from data/raw<br/>'
               'make serve   # dashboard on http://localhost:8501<br/>'
               'make api     # FastAPI on http://localhost:8000/docs<br/>'
               'python docs/build_report.py   # regenerate this report from results/', 'code'))
    F.append(P('Repository documents: docs/METHODOLOGY.md, docs/FEATURES.md, docs/RESULTS.md, docs/LIMITATIONS.md, docs/VIVA_QA.md, docs/DEMO_GUIDE.md, '
               'docs/REPORT_STRUCTURE.md, docs/FIX_PLAN.md, docs/AUDIT_SUMMARY.md, results/FINAL_RESULTS.md, results/experiments/experiments_summary.md.', 'small'))

    doc.build(F, canvasmaker=NumberedCanvas)
    print('wrote', OUT_REPORT)


def build_summary():
    FOOTER[0] = 'Multi-Asset Next-Day Return Prediction — Executive Summary'
    doc = BaseDocTemplate(OUT_SUMMARY, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.0 * cm, bottomMargin=2.3 * cm,
                          title='Executive Summary — Multi-Asset Next-Day Return Prediction', author='Alyan Shahid (teamlocalhost)')
    doc.addPageTemplates([PageTemplate(id='p', frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height)])])
    F = [P('Multi-Asset Next-Day Return Prediction — Bitcoin, Gold, Silver', 'title'), Spacer(1, 0.2 * cm),
         P('Final Year Project · Executive Summary for the Supervisor · Alyan Shahid (teamlocalhost), University of Lahore', 'subtitle'),
         Spacer(1, 0.3 * cm)]
    F += exec_summary_flowables()[1:]
    F.append(P('Key facts', 'h2'))
    F.append(table([
        ['Item', 'Final implementation'],
        ['Assets / horizon', 'Bitcoin, Gold (GC=F), Silver (SI=F); next trading day'],
        ['Target', 'log return ln(P_t+1 / P_t); price = P_t · exp(ŷ)'],
        ['Data', 'Yahoo Finance daily OHLCV 2018-01-01 → 2026-09-12 (BTC 3,147 / metals 2,156 usable days) + DXY, WTI, 10-y yield, S&P 500, VIX, Fear & Greed'],
        ['Features', '29 (BTC) / 28 (Gold) / 29 (Silver) stationary, backward-looking features incl. HAR/EWMA volatility and momentum; close-time rule (previous-day macro for the metals); raw price levels never used as inputs'],
        ['Split', f'train ≤ {TRAIN_END} · validation ≤ {VAL_END} · unseen test 2026-02-18 → 2026-09-12 (207 / 143 / 143 days), chronological, evaluated once'],
        ['Validation', f'{CV_FOLDS}-fold expanding-window walk-forward on train+val for tuning and model selection'],
        ['Models', 'Naive random walk, historical mean, ARIMA, Ridge, Random Forest, LightGBM, CatBoost, GRU, LSTM, stacked ensemble'],
        ['Served model', 'CatBoost for Bitcoin and Gold, GRU for Silver (walk-forward validation winners; all within 0.35 % of the random walk on validation)'],
        ['Metrics', 'MAE/RMSE/MAPE ($), RMSE/MAE/R² (return), directional accuracy + binomial p, Diebold–Mariano vs random walk, strategy backtest'],
        ['Stack', 'Python 3.9, pandas, scikit-learn, LightGBM, CatBoost, TensorFlow/Keras, statsmodels, Streamlit, FastAPI, Docker, pytest (34 cases), GitHub Actions'],
        ['Demo', 'Predict-a-Day on the unseen test period (predicted vs actual vs error), prediction history, live forecast with uncertainty band'],
    ], [3.0 * cm, 13.4 * cm]))
    F.append(P('Headline test-set results', 'h2'))
    d = [['Asset', 'Model', 'MAE ($)', 'RMSE ($)', 'MAPE %', 'R² (return)', 'Dir. acc. % (p)', 'DM p vs naive']]
    for a in ASSETS:
        for m in ('Naive', status[a]['primary_model']):
            r = row(a, m)
            d.append([a, NICE.get(m, m) + (' (served)' if m != 'Naive' else ''), f"{r['MAE_usd']:,.2f}", f"{r['RMSE_usd']:,.2f}", f"{r['MAPE_usd']:.2f}",
                      f"{r['R2_ret']:+.3f}", '—' if m == 'Naive' else f"{r['DirAcc_pct']:.1f} ({r['DirAcc_pvalue']:.2f})",
                      '—' if m == 'Naive' else f"{r['DM_pvalue']:.2f}"])
    t = table(d, [1.7 * cm, 3.6 * cm, 1.8 * cm, 1.9 * cm, 1.6 * cm, 2.0 * cm, 2.6 * cm, 2.2 * cm])
    t.setStyle(TableStyle([('ALIGN', (2, 1), (-1, -1), 'RIGHT')]))
    F.append(t)
    F.append(Paragraph('Unseen test period 2026-02-18 → 2026-09-12. Full 10-model tables: results/FINAL_RESULTS.md and the main report.', S['caption']))
    F.append(P('Reading the numbers', 'h2'))
    F.append(P('R² is reported on returns, where 0 means "as good as predicting the mean". The original project reported R² ≈ 0.95 on '
               'price levels — a value the random walk also achieves, which is why it was misleading. For Bitcoin and Gold the served models\' '
               'errors do not differ significantly from the random walk\'s (DM p = 0.74, 0.71). For Silver the served GRU is indistinguishable '
               'from the random walk (DM p = 0.99; 55 % direction, p = 0.12). An intermediate version showed 62 % for Silver; the final audit '
               'traced it to same-day macro features being used against a 13:30 ET futures settlement close, fixed the alignment and re-ran '
               'everything. On identical unseen days the improvement phase moved return-RMSE by −0.6 % (Gold), −0.4 % (Silver) and +0.1 % '
               '(Bitcoin) — inside noise; 2.9× more data made the models equal to the random walk, not better than it.'))
    F.append(P('<b>Current status:</b> implementation complete; pipeline reproducible; dashboard (with the unseen-day demonstration) and API '
               'verified running for all three assets; documentation, demo guide and viva notes prepared. <b>Recommended framing:</b> a '
               'leakage-audited forecasting benchmark and decision-support prototype whose predictions can be verified day by day, and which '
               'found and removed a subtle leak in its own intermediate results — not a market-beating predictor.', 'box'))
    doc.build(F, canvasmaker=NumberedCanvas)
    print('wrote', OUT_SUMMARY)


if __name__ == '__main__':
    build_report()
    build_summary()
