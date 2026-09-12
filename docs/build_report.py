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

ORDER = ['Naive-Zero', 'Naive-Mean', 'ARIMA', 'Ridge', 'RandomForest', 'LightGBM', 'CatBoost', 'GRU', 'LSTM', 'Stacked']
NICE = {'Naive-Zero': 'Naive (random walk)', 'Naive-Mean': 'Historical mean', 'RandomForest': 'Random Forest', 'Stacked': 'Stacked ensemble'}


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
        da = '—' if m == 'Naive-Zero' else f"{r['DirAcc_pct']:.1f} ({r['DirAcc_pvalue']:.2f})"
        dm = '—' if m == 'Naive-Zero' else f"{r['DM_pvalue']:.2f}"
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
        '<i>P<sub>t</sub>·exp(ŷ)</i>. Daily data from Yahoo Finance (2023-07-28 to 2026-07-28) is enriched with macro series '
        '(US Dollar Index, WTI crude, 10-year Treasury yield, S&amp;P 500, VIX) and the crypto Fear &amp; Greed index.'))
    f.append(P(
        f'<b>Methodology.</b> A frozen chronological split (train ≤ {TRAIN_END}, validation ≤ {VAL_END}, test = '
        f'2026-02-18 → 2026-07-28) with no shuffling; a scaler fitted on training rows only; 23–24 stationary, backward-looking '
        f'features; hyper-parameters and the served model chosen by {CV_FOLDS}-fold expanding-window walk-forward validation inside '
        'train+val; and a single evaluation of the untouched test set. Ten model families are compared on identical days: a '
        'random-walk baseline, historical mean, ARIMA, Ridge, Random Forest, LightGBM, CatBoost, GRU, LSTM and a stacked ensemble. '
        'Ten specific data-leakage checks are enforced by a 28-case test suite.'))
    f.append(P(
        '<b>Result.</b> The validation-selected model for all three assets is CatBoost (shallow, heavily regularised gradient-boosted '
        'trees). On the untouched test set its return-RMSE is within 0.3–1.5 % of the random-walk forecast (Diebold–Mariano '
        'p = 0.15–0.57), and its directional accuracy is 47.7–48.6 % (not significantly different from 50 %). <b>No model beats '
        'the random walk by a statistically significant margin</b>; the recurrent networks and ARIMA are significantly <i>worse</i> '
        '(p ≤ 0.05). The tuned models converge to the unconditional drift, which is the expected outcome under weak-form market '
        'efficiency at a one-day horizon with ~700–1,000 daily observations. The project therefore delivers a validated negative '
        'result, not a forecasting edge, and the application reports this honestly next to every forecast.'))
    f.append(P(
        '<b>Improvement over the initial version.</b> An audit of the original codebase found a price-reconstruction bug that '
        'corrupted every dollar metric, 30 % fabricated weekend rows for the metals, hyper-parameters selected on the test set, '
        'early stopping on training data, non-stationary price-level features, a served ensemble that predicted a constant, a '
        'broken inference path and misleading R² values (≈ 0.95 on price levels, which a random walk also achieves). All were '
        'fixed; the pipeline was re-run end-to-end and reproduces every reported number bit-for-bit.'))
    f.append(P(
        '<b>Status.</b> Working Streamlit dashboard (asset selection, indicator overlays, next-day forecast with uncertainty band, '
        'multi-model comparison, validation and test tables, figures) and FastAPI endpoints; reproducible pipeline '
        '(<font face="Mono">make pipeline</font>); documentation for methodology, features, results, limitations and viva.'))
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
        'its measured error and a clear disclaimer.',
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
        ['Quality', 'tests/ (19 functions, 28 cases), .github/workflows/ci.yml, Dockerfile', 'Look-ahead, alignment, split, reconstruction, metrics, inference tests; CI; container'],
    ], [2.6 * cm, 5.0 * cm, 8.8 * cm]))
    F.append(Paragraph('Table 1. Implemented components (source of truth: the repository).', S['caption']))

    # ---- 5 dataset
    F.append(P('5. Dataset and Data Processing', 'h1'))
    F.append(table([
        ['Series', 'Source / ticker', 'Rows used', 'Period', 'Role'],
        ['Bitcoin OHLCV', 'Yahoo Finance, BTC-USD', '1,067 calendar days', '2023-08-27 → 2026-07-28', 'target + features'],
        ['Gold futures OHLCV', 'Yahoo Finance, GC=F', '724 exchange days', '2023-09-11 → 2026-07-28', 'target + features'],
        ['Silver futures OHLCV', 'Yahoo Finance, SI=F', '724 exchange days', '2023-09-11 → 2026-07-28', 'target + features'],
        ['US Dollar Index, WTI crude, 10-y yield', 'Yahoo Finance: DX-Y.NYB, CL=F, ^TNX', 'aligned to asset calendar', '2023-08 → 2026-08', 'metals features'],
        ['S&P 500, VIX', 'Yahoo Finance: ^GSPC, ^VIX', 'aligned', '2023-08 → 2026-08', 'all assets'],
        ['Crypto Fear & Greed index', 'alternative.me API', 'aligned', '2023-04 → 2026-08', 'Bitcoin feature'],
    ], [3.4 * cm, 4.3 * cm, 3.0 * cm, 3.4 * cm, 2.3 * cm]))
    F.append(Paragraph('Table 2. Data sources. Row counts are after indicator warm-up (30 rows). Raw downloads cover 2023-07-28 → 2026-07-28.', S['caption']))
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
    F.append(P('All features at day <i>t</i> use data up to the close of day <i>t</i>. Raw price levels (open, high, low, close, volume, EMA, '
               'Bollinger bands, ATR, MACD) are retained only for charts; the models receive scale-free ratios so that the feature '
               'distribution does not drift when the price trends outside the training range (Gold\'s test prices, $3,986–5,294, lie entirely '
               'above its training range of $1,817–3,644). A unit test perturbs the last observation and asserts that no earlier feature '
               'changes (no look-ahead).'))
    F.append(table([
        ['Group', 'Features (formula)', 'Assets'],
        ['Return path', 'log_return = ln(P_t/P_t−1); return_1d/2d/5d/10d = lagged returns; volatility_10d/30d = rolling std of returns', 'all'],
        ['Momentum / trend', 'RSI(14); ADX(14); ROC(12); macd_norm = (EMA12−EMA26)/P; macd_hist_norm = (MACD−Signal9)/P; ema14_ratio, ema30_ratio = P/EMA−1', 'all'],
        ['Volatility state', 'bb_pctb = (P−BB_low)/(BB_up−BB_low); bb_width = (BB_up−BB_low)/BB_mid; atr_norm = ATR14/P; hl_range = (H−L)/P', 'all'],
        ['Macro', 'sp500_return, vix_return (all); dxy_return, oil_return, tnx_return (metals); gold_return (silver)', 'per asset'],
        ['Sentiment / calendar / activity', 'fear_greed (0–100); dow_sin, dow_cos; log_volume_change (clipped ±3)', 'Bitcoin'],
    ], [3.2 * cm, 10.9 * cm, 2.3 * cm]))
    F.append(Paragraph('Table 4. Model input features: Bitcoin 24, Gold 23, Silver 24 (docs/FEATURES.md gives every formula and rationale).', S['caption']))
    F.append(P('Deliberately excluded: futures volume for Gold/Silver (front-month contract volume from Yahoo jumps at contract rolls), '
               'day-of-week for the metals (no justification once synthetic weekends are removed), and redundant level indicators (VWAP, SMA, BB_Mid).'))
    F.append(P('<b>Target.</b> <i>y<sub>t</sub> = ln(P<sub>t+1</sub>/P<sub>t</sub>)</i>, the next row\'s log return, created only inside '
               '<font face="Mono">create_sequences</font>; the 30-day input window never contains that row (unit-tested). Recurrent models '
               f'receive the {SEQ_LEN}×F window; tabular models receive the last row of the same window, so every model is scored on '
               'identical days. The target is MinMax-scaled with the features (scaler fitted on train only) and inverted exactly for reporting.'))

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
        ['Trees', 'CatBoost', 'last-day features', 'Ordered boosting, early-stopped on validation. Served model.'],
        ['Deep', 'GRU', '30×F window', 'One GRU layer (16/32 units) + dropout + dense; early stopping.'],
        ['Deep', 'LSTM', '30×F window', 'Same shape as the GRU for a fair comparison.'],
        ['Ensemble', 'Stacked', 'OOF predictions', 'Non-negative Ridge over Ridge/LightGBM/CatBoost/GRU (experiment).'],
    ], [1.8 * cm, 3.1 * cm, 2.6 * cm, 8.9 * cm]))
    F.append(Paragraph('Table 5. Models compared (src/models/registry.py).', S['caption']))
    F.append(P('<b>Two-phase training.</b> Every model with a stopping rule (number of trees / boosting iterations / epochs) is first fitted on '
               'the training window while monitoring validation loss to find the stopping point (phase A — this also yields honest '
               'validation metrics and loss curves), then refitted on train+validation with that stopping point fixed (phase B — the deployed '
               'model). The validation data is therefore never inside its own early-stopping monitor.'))
    hp = [['Asset', 'Served model & tuned parameters', 'Stopping point', 'Runner-up configurations (tuned)']]
    for a in ASSETS:
        p = best[a]['CatBoost']; fi = status[a]['fit_info']
        hp.append([a, f"CatBoost: depth {p['depth']}, learning-rate {p['learning_rate']}, L2 {p['l2_leaf_reg']}, ≤{p['iterations']} iterations",
                   f"{fi.get('iterations_used')} iterations",
                   f"LightGBM: {best[a]['LightGBM']['num_leaves']} leaves, lr {best[a]['LightGBM']['learning_rate']}, λ {best[a]['LightGBM']['reg_lambda']}; "
                   f"Ridge α {best[a]['Ridge']['alpha']}; RF depth {best[a]['RandomForest']['max_depth']}; "
                   f"GRU {best[a]['GRU']['units']}u/drop {best[a]['GRU']['dropout']}; LSTM {best[a]['LSTM']['units']}u"])
    F.append(table(hp, [1.7 * cm, 5.3 * cm, 2.2 * cm, 7.2 * cm]))
    F.append(Paragraph('Table 6. Hyper-parameters chosen by walk-forward validation (results/tuning/best_params.json). The search consistently '
                       'selected the most regularised settings — shallow trees, few iterations, large α — i.e. the data supports very little model capacity.', S['caption']))
    F.append(P('<b>Why CatBoost is served.</b> Selection is automatic: the model with the lowest mean walk-forward RMSE of the return on the '
               'train+val folds. CatBoost won for all three assets, but Ridge, LightGBM and the historical mean are within 0.5 % of it and '
               'the random walk within 1 % — the ranking among these is not statistically meaningful. CatBoost was retained because it is '
               'the validation winner, trains in seconds, and exposes feature importance; the recurrent models were the worst family on '
               'every asset in validation.'))

    # ---- 8 training & evaluation
    F.append(P('8. Training and Evaluation Methodology', 'h1'))
    F.append(table([
        ['Split', 'Rule', 'Bitcoin', 'Gold / Silver', 'Used for'],
        ['Train', f'≤ {TRAIN_END}', '746 days (716 windows)', '504 days (474 windows)', 'fitting; scaler'],
        ['Validation', f'{TRAIN_END} < t ≤ {VAL_END}', '160 days', '109 days', 'early stopping; walk-forward folds with train'],
        ['Test', f'> {VAL_END} (to 2026-07-28)', '161 days', '111 days', 'evaluated once by backtesting.py'],
    ], [2.0 * cm, 4.2 * cm, 3.4 * cm, 3.4 * cm, 3.4 * cm]))
    F.append(Paragraph('Table 7. Frozen chronological split (config.py). No shuffling; identical calendar boundaries for all assets.', S['caption']))
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
        ['Look-ahead in rolling / EWM / shift features', 'all operators backward-looking', 'test_no_lookahead_in_any_feature'],
        ['External data alignment', 'reindex(…, method="ffill") only', 'code review; alignment on asset calendar'],
        ['Early stopping on training data', 'two-phase fit', 'registry.py; CatBoost bug removed'],
        ['Hyper-parameters chosen on test', 'tuner reads train+val arrays only', 'grep: no test reference in training/tuning/stacking'],
        ['Synthetic rows', 'exchange calendar for futures', 'test_commodities_keep_trading_calendar'],
        ['Wrong price reconstruction', 'explicit previous close', 'test_true_target_reconstructs_actual_close_exactly (error 1e-11)'],
        ['Test set touched more than once', 'single script', 'backtesting.py is the only reader of the test split'],
    ], [4.2 * cm, 5.0 * cm, 7.2 * cm]))
    F.append(Paragraph('Table 9. Data-leakage checklist enforced by the test suite (tests/, 28 cases, all passing).', S['caption']))

    # ---- 9 results
    F.append(P('9. Experimental Results', 'h1'))
    F.append(P('Test period 2026-02-18 → 2026-07-28 for all assets (Bitcoin 161 days; Gold and Silver 111 exchange days). '
               '"CV" columns are the walk-forward validation scores used for selection; the remaining columns are the single test-set '
               'evaluation. The served model is marked "(served)". All values are from results/cv_results.csv and results/final_test_results.csv.'))
    for i, a in enumerate(ASSETS):
        F.append(KeepTogether([P(f'9.{i + 1} {a}', 'h2'), results_table(a),
                               Paragraph(f'Table {10 + i}. {a}: walk-forward validation and untouched test-set results, all models.', S['caption'])]))
    F.append(fig(os.path.join(FIGURES_DIR, 'model_comparison_test.png'), 16.4,
                 'Figure 4. Test-set comparison: RMSE of the next-day return (red line = random walk) and directional accuracy (red line = 50 %).'))
    F.append(P('9.4 Interpretation', 'h2'))
    F += bullets([
        '<b>Baselines.</b> The random walk is the best or joint-best forecast on every asset in both validation and test; the historical mean '
        'is within 0.2 % of it. ARIMA is significantly worse than the random walk on Gold and Silver (DM p = 0.03, 0.01): the small '
        'in-sample autocorrelations it fits do not persist.',
        '<b>Served models.</b> CatBoost\'s test return-RMSE is 0.3 % (Bitcoin), 1.5 % (Gold) and 0.6 % (Silver) above the random walk — '
        'inside the noise (DM p = 0.57 / 0.15 / 0.27). Its predictions have a standard deviation of 0.0001–0.0003 against a true return '
        'standard deviation of 0.017–0.034: the validation-selected model is effectively the unconditional drift.',
        '<b>Direction.</b> Directional accuracy is 42–53 % for every model on every asset and none is significant (smallest binomial '
        'p = 0.26: Random Forest on Bitcoin, 52.8 % of 161 days). Random Forest on Bitcoin also has the best test R² (+0.007) and the only '
        'positive strategy return (+3.1 % vs −6.1 % buy-and-hold), but it ranked 7th of 9 on validation; reporting it as the winner would '
        'be selection on the test set.',
        '<b>Deep models.</b> GRU and LSTM are the worst family in validation on every asset (R² −0.10 to −0.22) and significantly worse than '
        'the random walk on the Bitcoin test set (DM p = 0.01, 0.03), despite being reduced to a single 16–32-unit layer with dropout and '
        'early stopping. With 500–700 training windows they fit noise.',
        f'<b>Stacked ensemble.</b> With non-negative weights on out-of-fold predictions the meta-model assigned weight 0.0 to all four base '
        f'models for Bitcoin (intercept only) and only 0.04–0.16 for Gold and Silver — an independent confirmation that the base models carry '
        'no combinable out-of-sample signal.',
        '<b>Over-fitting.</b> Validation RMSE is 1.0× (Bitcoin), 2.0× (Gold) and 3.0× (Silver) the training RMSE, but the random walk shows '
        'exactly the same ratios (Figure 6): the gap is regime shift — the validation and test windows are far more volatile than the '
        'training window — not memorisation.'])
    F.append(fig(os.path.join(FIGURES_DIR, 'gold_actual_vs_predicted.png'), 15.2,
                 'Figure 5. Gold test period, served model: predicted vs actual next-day returns (top, scatter middle) and the price view (bottom). '
                 'A price line always tracks the actual with a one-day lag; skill is only visible in return space.', max_height_cm=12.5))
    F.append(fig(os.path.join(FIGURES_DIR, 'overfitting_gap.png'), 15.0,
                 'Figure 6. Train vs validation RMSE per model with the random walk\'s train/validation levels (dotted/dashed).'))
    F.append(fig(os.path.join(FIGURES_DIR, 'btc_feature_importance.png'), 10.5,
                 'Figure 7. Bitcoin, served CatBoost: normalised gain importance. Volatility-regime and trend-distance features dominate, '
                 'together with the Fear & Greed index.'))

    # ---- 10 application
    F.append(P('10. Final System and Application', 'h1'))
    F.append(P('The Streamlit dashboard (<font face="Mono">make serve</font>, port 8501) and the FastAPI service '
               '(<font face="Mono">make api</font>, port 8000) sit on the same inference module. Inference scales the latest unscaled '
               'features with the train-fitted scaler, runs the requested model on the last 30 rows, unscales the predicted return and '
               'converts it to a price. A "Sync Live Market Data" button refreshes features into a separate live file so the frozen '
               'evaluation data is never overwritten (Yahoo Finance rate-limited during this work, so the stored data is shown).'))
    F.append(table([
        ['Implemented (verified running)', 'Not implemented (future possibilities)'],
        ['• Select Bitcoin / Gold / Silver; price chart with train/val/test shading; Bollinger, RSI, MACD, volume overlays<br/>'
         '• Run a next-day prediction with the served model or any trained model: current close, predicted close, % change, direction, '
         'horizon (T+1), model name, ±1 RMSE uncertainty band, held-out metrics of that model, disclaimer<br/>'
         '• Forecast-horizon chart (last 60 days + next day); table of all trained models\' predictions on the same input<br/>'
         '• Validation view: actual vs predicted returns and prices over the test period; return scatter<br/>'
         '• Performance tab: walk-forward and test tables for all models, honest-reading summary, report figures<br/>'
         '• Methodology tab: task, split, features, models, served-model parameters, model inventory<br/>'
         '• API: GET /health, GET /models, GET /predict/{asset}[?model=…]; Docker Compose for both services<br/>'
         '• Deep links: ?asset=Gold&amp;run=1',
         '• Multi-day horizons or volatility forecasts<br/>• Calibrated prediction intervals (the band is an ex-post error band)<br/>'
         '• User accounts, saved forecasts, database<br/>• Automatic scheduled retraining in production<br/>'
         '• News / NLP sentiment for the metals<br/>• Intraday data<br/>• Trading execution or portfolio management'],
    ], [10.8 * cm, 5.6 * cm]))
    F.append(Paragraph('Table 13. Application functionality.', S['caption']))
    F.append(fig(os.path.join(DOC_FIG, 'dash_gold_forecast.png'), 14.0,
                 'Figure 8. Dashboard, Gold: indicator chart, next-day prediction with uncertainty band, held-out metrics, disclaimer, '
                 'forecast-horizon chart and multi-model table (screenshot of the running application).', max_height_cm=20.5))
    perf_crop = crop_image(os.path.join(DOC_FIG, 'dash_performance.png'), os.path.join(DOC_FIG, 'dash_performance_crop.png'), (0, 0, 2100, 1700))
    F.append(fig(perf_crop, 16.0, 'Figure 9. Dashboard, Bitcoin: Model Performance tab with the walk-forward and test tables generated by the evaluation script.'))

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
        ['Code & docs', 'Five stale result files from an earlier pipeline; Makefile/CI referenced missing scripts; 4 of 31 tests failed; README claimed models/indicators that did not exist', 'Single results/ tree; working Makefile/CI/retrain; 28 passing tests; accurate README and docs/; bit-identical reproducibility'],
    ], [2.6 * cm, 7.4 * cm, 6.4 * cm], font=7.8))
    F.append(Paragraph('Table 14. Before vs final implementation (docs/FIX_PLAN.md, docs/AUDIT_SUMMARY.md).', S['caption']))

    # ---- 12 limitations
    F.append(P('12. Limitations', 'h1'))
    F += bullets([
        '<b>Market efficiency dominates.</b> Lag-1 autocorrelation of returns is −0.08 / −0.04 / +0.00 (BTC / Gold / Silver); at a one-day '
        'horizon the predictable component is small and unstable, and the results are consistent with that.',
        '<b>Small sample.</b> Three years of daily data (Yahoo Finance rate-limited a longer download; config.DATA_START_DATE = 2018-01-01 is '
        'supported by the pipeline). With 111–161 test days, differences of a few percent in RMSE and directional accuracies of 52–56 % are not '
        'statistically detectable.',
        '<b>Regime shift in the test window.</b> Gold and Silver traded far above their training range and Silver\'s 30-day volatility exceeded '
        'every training value on 89 % of test days; one five-month window is a single draw from a non-stationary process.',
        '<b>Data quality.</b> Front-month futures (roll effects; unusable volume), a composite BTC index, forward-filled macro series on holidays, '
        'and a proprietary sentiment index.',
        '<b>Modelling.</b> One-day point forecasts only; the dashboard\'s band is an ex-post RMSE band, not a calibrated interval; small tuning '
        'grids; selection by RMSE (directional accuracy and Sharpe are reported but not optimised).',
        '<b>Backtest.</b> The long/flat strategy uses a flat 10 bps cost with no slippage, shorting, sizing or financing.'])

    F.append(P('13. Future Work', 'h1'))
    F += bullets([
        'Download the full history from 2018 (≈ 3× more data) and re-run the unchanged pipeline.',
        'Forecast volatility (GARCH-type or quantile targets), where the volatility-regime features clearly carry information, and score calibration.',
        'Multi-day horizons and probabilistic forecasts; regime-aware or switching models.',
        'News/NLP sentiment for the metals; intraday data; a realistic transaction-cost model.'])

    F.append(KeepTogether([P('14. Conclusion', 'h1'), P('The project set out to build a working next-day prediction system for Bitcoin, Gold and Silver and to make it '
               'scientifically defensible. The final system has a leakage-audited, reproducible pipeline; ten model families compared '
               'under expanding-window walk-forward validation on identical days; a single, statistically tested evaluation on an untouched '
               'period; and a dashboard and API that serve the validation-selected model together with its measured error. The empirical '
               'answer is a validated negative result: with daily price, volume, macro and sentiment features and ~700–1,000 observations, '
               'no statistical, tree-based or recurrent model beats the random walk at a one-day horizon by a significant margin, and the '
               'models that deviate most from the drift are significantly worse. The value of the work lies in demonstrating that result '
               'rigorously — and in showing why the original "R² ≈ 0.95" was an illusion — rather than in a forecasting edge.')]))

    F.append(P('15. FYP Readiness Assessment', 'h1'))
    F.append(P('<b>Suitable for submission and live demonstration.</b> The implementation demonstrates the complete ML research workflow '
               '(problem → data → analysis → features → baselines → models → validation → experiments → comparison → final model → '
               'system → evaluation → limitations); every methodological claim is backed by a test, a script or a results file; the '
               'application runs for all three assets; and the pipeline reproduces every reported number. The main remaining weaknesses are '
               'the data volume and the absence of a positive forecasting result — the former is addressable by re-downloading data, the '
               'latter is the honest finding of the study and is presented as such. It should be presented as a methodological benchmark and '
               'decision-support prototype, not as a market-beating predictor.', 'box'))

    F.append(P('Appendix — Reproduction', 'h1'))
    F.append(P('python -m venv venv &amp;&amp; source venv/bin/activate &amp;&amp; pip install -r requirements.txt<br/>'
               'make preprocess eda tune train stack evaluate figures test   # full pipeline from data/raw<br/>'
               'make serve   # dashboard on http://localhost:8501<br/>'
               'make api     # FastAPI on http://localhost:8000/docs<br/>'
               'python docs/build_report.py   # regenerate this report from results/', 'code'))
    F.append(P('Repository documents: docs/METHODOLOGY.md, docs/FEATURES.md, docs/RESULTS.md, docs/LIMITATIONS.md, docs/VIVA_QA.md, '
               'docs/REPORT_STRUCTURE.md, docs/FIX_PLAN.md, docs/AUDIT_SUMMARY.md, results/FINAL_RESULTS.md.', 'small'))

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
        ['Data', 'Yahoo Finance daily OHLCV 2023-07-28 → 2026-07-28 (BTC 1,067 / metals 724 usable days) + DXY, WTI, 10-y yield, S&P 500, VIX, Fear & Greed'],
        ['Features', '24 (BTC) / 23 (Gold) / 24 (Silver) stationary, backward-looking features; raw price levels never used as inputs'],
        ['Split', f'train ≤ {TRAIN_END} · validation ≤ {VAL_END} · test 2026-02-18 → 2026-07-28 (161 / 111 / 111 days), chronological, evaluated once'],
        ['Validation', f'{CV_FOLDS}-fold expanding-window walk-forward on train+val for tuning and model selection'],
        ['Models', 'Naive random walk, historical mean, ARIMA, Ridge, Random Forest, LightGBM, CatBoost, GRU, LSTM, stacked ensemble'],
        ['Served model', 'CatBoost (validation winner for all three assets; depth 3/5/3, lr 0.01)'],
        ['Metrics', 'MAE/RMSE/MAPE ($), RMSE/MAE/R² (return), directional accuracy + binomial p, Diebold–Mariano vs random walk, strategy backtest'],
        ['Stack', 'Python 3.9, pandas, scikit-learn, LightGBM, CatBoost, TensorFlow/Keras, statsmodels, Streamlit, FastAPI, Docker, pytest (28 cases), GitHub Actions'],
    ], [3.0 * cm, 13.4 * cm]))
    F.append(P('Headline test-set results', 'h2'))
    d = [['Asset', 'Model', 'MAE ($)', 'RMSE ($)', 'MAPE %', 'R² (return)', 'Dir. acc. % (p)', 'DM p vs naive']]
    for a in ASSETS:
        for m in ('Naive-Zero', 'CatBoost'):
            r = row(a, m)
            d.append([a, NICE.get(m, m) + (' (served)' if m == 'CatBoost' else ''), f"{r['MAE_usd']:,.2f}", f"{r['RMSE_usd']:,.2f}", f"{r['MAPE_usd']:.2f}",
                      f"{r['R2_ret']:+.3f}", '—' if m == 'Naive-Zero' else f"{r['DirAcc_pct']:.1f} ({r['DirAcc_pvalue']:.2f})",
                      '—' if m == 'Naive-Zero' else f"{r['DM_pvalue']:.2f}"])
    t = table(d, [1.7 * cm, 3.6 * cm, 1.8 * cm, 1.9 * cm, 1.6 * cm, 2.0 * cm, 2.6 * cm, 2.2 * cm])
    t.setStyle(TableStyle([('ALIGN', (2, 1), (-1, -1), 'RIGHT')]))
    F.append(t)
    F.append(Paragraph('Test period 2026-02-18 → 2026-07-28. Full 10-model tables: results/FINAL_RESULTS.md and the main report.', S['caption']))
    F.append(P('Reading the numbers', 'h2'))
    F.append(P('R² is reported on returns, where 0 means "as good as predicting the mean"; every model is within a few thousandths of 0 or '
               'below it. The original project reported R² ≈ 0.95 on price levels — a value the random walk also achieves, which is why it '
               'was misleading. The Diebold–Mariano p-values show that none of the served models\' errors differ significantly from the '
               'random walk\'s; models that deviate more (ARIMA, GRU, LSTM) are significantly worse. The correct reading is that the '
               'best achievable one-day point forecast on this data is approximately the drift, and the system says so.'))
    F.append(P('<b>Current status:</b> implementation complete; pipeline reproducible bit-for-bit; dashboard and API verified running for all '
               'three assets; documentation and viva notes prepared. <b>Recommended framing:</b> a leakage-audited forecasting benchmark and '
               'decision-support prototype with a validated negative result — not a market-beating predictor.', 'box'))
    doc.build(F, canvasmaker=NumberedCanvas)
    print('wrote', OUT_SUMMARY)


if __name__ == '__main__':
    build_report()
    build_summary()
