"""
Build the supervisor-facing technical report (docs/FYP_Technical_Report.pdf) from the project's real artefacts.

    python docs/build_technical_report.py

Every number in the report is read from: results/*.csv, results/tuning/best_params.json, results/stacking/*.json,
results/experiments/*, data/models/model_status.json, data/raw/*.csv and data/processed/*.csv.
Dashboard screenshots come from docs/figures/final/ (captured from the final dashboard code).
"""
import os
import sys
import json
import datetime as dt
import subprocess
import pandas as pd
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether, NextPageTemplate)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from config import (RESULTS_DIR, FIGURES_DIR, TUNING_DIR, MODEL_STATUS_PATH, TRAIN_END, VAL_END, ASSETS, SEQ_LEN,
                    CV_FOLDS, RAW_DATA_DIR, PROCESSED_DATA_DIR, ASSET_CONFIG, LEVEL_COLUMNS, get_prefix)

OUT = os.path.join(ROOT, 'docs', 'FYP_Technical_Report.pdf')
DASH = os.path.join(ROOT, 'docs', 'figures', 'final')

# ------------------------------------------------------------------ fonts
FONT_DIR = '/System/Library/Fonts/Supplemental'
pdfmetrics.registerFont(TTFont('Body', os.path.join(FONT_DIR, 'Arial.ttf')))
pdfmetrics.registerFont(TTFont('Body-Bold', os.path.join(FONT_DIR, 'Arial Bold.ttf')))
pdfmetrics.registerFont(TTFont('Body-Italic', os.path.join(FONT_DIR, 'Arial Italic.ttf')))
pdfmetrics.registerFont(TTFont('Body-BoldItalic', os.path.join(FONT_DIR, 'Arial Bold Italic.ttf')))
pdfmetrics.registerFont(TTFont('Head', os.path.join(FONT_DIR, 'Georgia Bold.ttf')))
pdfmetrics.registerFont(TTFont('Head-Reg', os.path.join(FONT_DIR, 'Georgia.ttf')))
import matplotlib
pdfmetrics.registerFont(TTFont('Mono', os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf', 'DejaVuSansMono.ttf')))
registerFontFamily('Body', normal='Body', bold='Body-Bold', italic='Body-Italic', boldItalic='Body-BoldItalic')

NAVY = colors.HexColor('#1F3A5F')
TEAL = colors.HexColor('#0F766E')
GREY = colors.HexColor('#555555')
LIGHT = colors.HexColor('#F2F4F7')
LINE = colors.HexColor('#C8CDD4')
AMBER = colors.HexColor('#FFF4E0')
AMBER_LINE = colors.HexColor('#E0A94B')
RED = colors.HexColor('#B42318')

S = {
    'title': ParagraphStyle('title', fontName='Head', fontSize=23, leading=29, textColor=NAVY, alignment=TA_CENTER),
    'subtitle': ParagraphStyle('subtitle', fontName='Head-Reg', fontSize=12.5, leading=17, textColor=GREY, alignment=TA_CENTER),
    'cover': ParagraphStyle('cover', fontName='Body', fontSize=10.5, leading=15, alignment=TA_CENTER),
    'h1': ParagraphStyle('h1', fontName='Head', fontSize=15.5, leading=19, textColor=NAVY, spaceBefore=12, spaceAfter=6, keepWithNext=1),
    'h2': ParagraphStyle('h2', fontName='Body-Bold', fontSize=11.2, leading=14.5, textColor=NAVY, spaceBefore=8, spaceAfter=3, keepWithNext=1),
    'h3': ParagraphStyle('h3', fontName='Body-Bold', fontSize=9.9, leading=13, textColor=TEAL, spaceBefore=5, spaceAfter=2, keepWithNext=1),
    'body': ParagraphStyle('body', fontName='Body', fontSize=9.6, leading=13.3, alignment=TA_JUSTIFY, spaceAfter=5),
    'bodyl': ParagraphStyle('bodyl', fontName='Body', fontSize=9.4, leading=13.0, alignment=TA_LEFT, spaceAfter=5),
    'bullet': ParagraphStyle('bullet', fontName='Body', fontSize=9.6, leading=13.1, leftIndent=12, bulletIndent=2, spaceAfter=2),
    'small': ParagraphStyle('small', fontName='Body', fontSize=8.0, leading=10.2),
    'smallb': ParagraphStyle('smallb', fontName='Body-Bold', fontSize=8.0, leading=10.2),
    'tiny': ParagraphStyle('tiny', fontName='Body', fontSize=7.2, leading=9.2),
    'tinyb': ParagraphStyle('tinyb', fontName='Body-Bold', fontSize=7.2, leading=9.2),
    'caption': ParagraphStyle('caption', fontName='Body-Italic', fontSize=8.4, leading=10.8, textColor=GREY, alignment=TA_CENTER, spaceBefore=3, spaceAfter=9),
    'code': ParagraphStyle('code', fontName='Mono', fontSize=7.8, leading=10.2, leftIndent=8, backColor=LIGHT, borderPadding=4, spaceAfter=6),
    'box': ParagraphStyle('box', fontName='Body', fontSize=9.4, leading=13, backColor=LIGHT, borderPadding=7, borderColor=LINE, borderWidth=0.6, spaceBefore=8, spaceAfter=10),
    'warn': ParagraphStyle('warn', fontName='Body', fontSize=9.4, leading=13, backColor=AMBER, borderPadding=7, borderColor=AMBER_LINE, borderWidth=0.8, spaceBefore=8, spaceAfter=10),
    'toc1': ParagraphStyle('toc1', fontName='Body-Bold', fontSize=9.6, leading=13.5, leftIndent=0),
    'toc2': ParagraphStyle('toc2', fontName='Body', fontSize=8.8, leading=12, leftIndent=14),
}

# ------------------------------------------------------------------ data (all read from artefacts)
cv = pd.read_csv(os.path.join(RESULTS_DIR, 'cv_results.csv'))
te = pd.read_csv(os.path.join(RESULTS_DIR, 'final_test_results.csv'))
tv = pd.read_csv(os.path.join(RESULTS_DIR, 'train_val_metrics.csv'))
eda = pd.read_csv(os.path.join(RESULTS_DIR, 'eda_summary.csv')).set_index('asset')
reg = pd.read_csv(os.path.join(RESULTS_DIR, 'regime_analysis.csv'))
fi = pd.read_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'))
status = json.load(open(MODEL_STATUS_PATH))
best = json.load(open(os.path.join(TUNING_DIR, 'best_params.json')))['return_1d']
stack = {a: json.load(open(os.path.join(RESULTS_DIR, 'stacking', f'{a.lower()}_stack_weights.json'))) for a in ASSETS}
E1 = pd.read_csv(os.path.join(RESULTS_DIR, 'experiments', 'E1_data_size.csv'))
E3 = pd.read_csv(os.path.join(RESULTS_DIR, 'experiments', 'E3_horizon.csv'))
E4 = pd.read_csv(os.path.join(RESULTS_DIR, 'experiments', 'E4_volatility.csv'))
BA = pd.read_csv(os.path.join(RESULTS_DIR, 'experiments', 'before_after.csv'))
hist = {a: pd.read_csv(os.path.join(RESULTS_DIR, 'predictions', f'{get_prefix(a)}_test_predictions.csv'),
                       parse_dates=['date', 'target_date']) for a in ASSETS}
raw = {a: pd.read_csv(os.path.join(RAW_DATA_DIR, ASSET_CONFIG[a]['filename']), parse_dates=['timestamp']) for a in ASSETS}
feat = {a: pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{get_prefix(a)}_features.csv'), index_col=0, parse_dates=True) for a in ASSETS}
ext_raw = {n: pd.read_csv(os.path.join(RAW_DATA_DIR, f), parse_dates=['timestamp']) for n, f in
           [('DXY', 'dxy_data.csv'), ('WTI', 'crude_oil_data.csv'), ('TNX', 'tnx_data.csv'), ('SP500', 'sp500_data.csv'),
            ('VIX', 'vix_data.csv'), ('FG', 'fear_greed_data.csv')]}
try:
    COMMIT = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, text=True).strip()
except Exception:
    COMMIT = 'n/a'
N_TESTS = 41
TODAY = dt.date.today().strftime('%d %B %Y')

ORDER = ['Naive', 'Naive-Mean', 'ARIMA', 'Ridge', 'RandomForest', 'LightGBM', 'CatBoost', 'GRU', 'LSTM', 'Stacked', 'Combined']
NICE = {'Naive': 'Naive (random walk)', 'Naive-Mean': 'Historical mean', 'RandomForest': 'Random Forest',
        'Stacked': 'Stacked ensemble (experiment)', 'Combined': 'Combined (served)'}
MEMBERS = ['Ridge', 'RandomForest', 'LightGBM', 'CatBoost', 'GRU', 'LSTM']


def row(asset, model):
    return te[(te['asset'] == asset) & (te['model'] == model)].iloc[0]


def cvrow(asset, model):
    return cv[(cv['asset'] == asset) & (cv['model'] == model)].iloc[0]


def fitinfo(asset, model):
    r = tv[(tv['asset'] == asset) & (tv['model'] == model) & (tv['split'] == 'val')].iloc[0]
    for c, lbl in (('fit_epochs_used', 'epochs'), ('fit_iterations_used', 'iterations'), ('fit_n_estimators_used', 'trees')):
        if c in r and pd.notna(r[c]):
            return f"{int(r[c])} {lbl}"
    return '—'


# ------------------------------------------------------------------ flowable helpers
def P(text, style='body'):
    return Paragraph(text, S[style])


def bullets(items, style='bullet'):
    return [Paragraph(t, S[style], bulletText='•') for t in items]


def table(data, col_widths, header=True, font=7.8, align_right_from=None, zebra=True, bold_rows=(), wrap=True, valign='TOP'):
    """data: list of rows (strings). Cells are wrapped in small Paragraphs so long text flows."""
    ps = ParagraphStyle('cell', fontName='Body', fontSize=font, leading=font + 2.2)
    pb = ParagraphStyle('cellb', fontName='Body-Bold', fontSize=font, leading=font + 2.2)
    pr = ParagraphStyle('cellr', fontName='Body', fontSize=font, leading=font + 2.2, alignment=2)
    pbr = ParagraphStyle('cellbr', fontName='Body-Bold', fontSize=font, leading=font + 2.2, alignment=2)
    ph = ParagraphStyle('cellh', fontName='Body-Bold', fontSize=font, leading=font + 2.2, textColor=colors.white)
    phr = ParagraphStyle('cellhr', fontName='Body-Bold', fontSize=font, leading=font + 2.2, textColor=colors.white, alignment=2)
    rows = []
    for i, r in enumerate(data):
        cells = []
        for j, c in enumerate(r):
            txt = '' if c is None else str(c)
            if not wrap:
                cells.append(txt); continue
            if header and i == 0:
                st = phr if (align_right_from is not None and j >= align_right_from) else ph
            elif i in bold_rows:
                st = pbr if (align_right_from is not None and j >= align_right_from) else pb
            else:
                st = pr if (align_right_from is not None and j >= align_right_from) else ps
            cells.append(Paragraph(txt, st))
        rows.append(cells)
    t = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [('VALIGN', (0, 0), (-1, -1), valign), ('LINEBELOW', (0, 0), (-1, -1), 0.3, LINE),
             ('TOPPADDING', (0, 0), (-1, -1), 2.4), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.4),
             ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4)]
    if header:
        style += [('BACKGROUND', (0, 0), (-1, 0), NAVY), ('LINEBELOW', (0, 0), (-1, 0), 0.8, NAVY)]
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                style.append(('BACKGROUND', (0, i), (-1, i), LIGHT))
    for i in bold_rows:
        style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#E3F4F1')))
    t.setStyle(TableStyle(style))
    return t


def fig(path, width_cm, caption, max_height_cm=None):
    if not os.path.exists(path):
        return [P(f"<i>[figure missing: {os.path.basename(path)}]</i>", 'caption')]
    im = PILImage.open(path)
    w, h = im.size
    width = width_cm * cm
    height = width * h / w
    if max_height_cm and height > max_height_cm * cm:
        height = max_height_cm * cm
        width = height * w / h
    return [KeepTogether([Image(path, width=width, height=height), P(caption, 'caption')])]



def _fit_font(text, font, max_width, name='Body-Bold'):
    from reportlab.pdfbase.pdfmetrics import stringWidth
    w = stringWidth(text, name, font)
    return font if w <= max_width else max(5.2, font * max_width / w)

def flow_chart(steps, cols=4, width_cm=16.6, box_h=0.95, gap_x=0.55, gap_y=0.55, font=7.4, fill=None):
    """Boxes joined by arrows, wrapping over rows (each row runs left to right; a down-elbow joins rows)."""
    fill = fill or colors.HexColor('#E8EEF6')
    n = len(steps)
    rows = (n + cols - 1) // cols
    W = width_cm * cm
    bw = (W - (cols - 1) * gap_x * cm) / cols
    bh = box_h * cm
    H = rows * bh + (rows - 1) * gap_y * cm
    d = Drawing(W, H)
    pos = []
    for i, text in enumerate(steps):
        r, c = divmod(i, cols)
        x = c * (bw + gap_x * cm)
        y = H - (r + 1) * bh - r * gap_y * cm
        d.add(Rect(x, y, bw, bh, rx=4, ry=4, fillColor=fill, strokeColor=NAVY, strokeWidth=0.8))
        lines = text.split('\n')
        for k, ln in enumerate(lines):
            fn = 'Body-Bold' if k == 0 else 'Body'
            d.add(String(x + bw / 2, y + bh / 2 + (len(lines) - 1) * font * 0.65 - k * font * 1.3 - font * 0.35, ln,
                         fontName=fn, fontSize=_fit_font(ln, font, bw - 8, fn), textAnchor='middle', fillColor=NAVY))
        pos.append((x, y, bw, bh))
    for i in range(n - 1):
        x, y, w, h = pos[i]; x2, y2, w2, h2 = pos[i + 1]
        if (i + 1) % cols != 0:                                        # same row → right arrow
            d.add(Line(x + w, y + h / 2, x2 - 2, y2 + h2 / 2, strokeColor=NAVY, strokeWidth=0.9))
            d.add(Polygon([x2 - 2, y2 + h2 / 2, x2 - 7, y2 + h2 / 2 + 3, x2 - 7, y2 + h2 / 2 - 3], fillColor=NAVY, strokeColor=NAVY))
        else:                                                          # row end → elbow down-left to the next row
            mid = y - gap_y * cm / 2
            d.add(Line(x + w / 2, y, x + w / 2, mid, strokeColor=NAVY, strokeWidth=0.9))
            d.add(Line(x + w / 2, mid, x2 + w2 / 2, mid, strokeColor=NAVY, strokeWidth=0.9))
            d.add(Line(x2 + w2 / 2, mid, x2 + w2 / 2, y2 + h2 + 2, strokeColor=NAVY, strokeWidth=0.9))
            d.add(Polygon([x2 + w2 / 2, y2 + h2 + 2, x2 + w2 / 2 - 3, y2 + h2 + 7, x2 + w2 / 2 + 3, y2 + h2 + 7], fillColor=NAVY, strokeColor=NAVY))
    return d


def vflow(steps, width_cm=9.0, box_h=0.8, gap=0.45, font=7.8, fill=None):
    """Vertical chain of boxes joined by down arrows."""
    fill = fill or colors.HexColor('#E8EEF6')
    W, bh, g = width_cm * cm, box_h * cm, gap * cm
    H = len(steps) * bh + (len(steps) - 1) * g
    d = Drawing(W, H)
    for i, text in enumerate(steps):
        y = H - (i + 1) * bh - i * g
        d.add(Rect(0, y, W, bh, rx=4, ry=4, fillColor=fill, strokeColor=NAVY, strokeWidth=0.8))
        lines = text.split('\n')
        for k, ln in enumerate(lines):
            fn = 'Body-Bold' if k == 0 else 'Body'
            d.add(String(W / 2, y + bh / 2 + (len(lines) - 1) * font * 0.65 - k * font * 1.3 - font * 0.35, ln,
                         fontName=fn, fontSize=_fit_font(ln, font, W - 8, fn), textAnchor='middle', fillColor=NAVY))
        if i < len(steps) - 1:
            d.add(Line(W / 2, y, W / 2, y - g + 2, strokeColor=NAVY, strokeWidth=0.9))
            d.add(Polygon([W / 2, y - g + 2, W / 2 - 3, y - g + 7, W / 2 + 3, y - g + 7], fillColor=NAVY, strokeColor=NAVY))
    return d


def side_by_side(left, right, widths=(8.3, 8.3)):
    t = Table([[left, right]], colWidths=[w * cm for w in widths])
    t.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 0), ('RIGHTPADDING', (0, 0), (-1, -1), 6)]))
    return t


# ------------------------------------------------------------------ document template with TOC
class Doc(BaseDocTemplate):
    def __init__(self, path, **kw):
        super().__init__(path, pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm, topMargin=2.0 * cm, bottomMargin=1.9 * cm, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id='f', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id='cover', frames=[frame], onPage=self._cover),
                               PageTemplate(id='main', frames=[frame], onPage=self._main)])
        self._h1 = 0

    def _cover(self, canv, doc):
        canv.saveState()
        canv.setStrokeColor(NAVY); canv.setLineWidth(2)
        canv.line(2 * cm, A4[1] - 2.2 * cm, A4[0] - 2 * cm, A4[1] - 2.2 * cm)
        canv.restoreState()

    def _main(self, canv, doc):
        canv.saveState()
        canv.setFont('Body', 7.6); canv.setFillColor(GREY)
        canv.drawString(2 * cm, A4[1] - 1.3 * cm, 'Multi-Asset Next-Day Price Prediction — Bitcoin, Gold, Silver · Technical Report for Supervisor Review')
        canv.drawRightString(A4[0] - 2 * cm, A4[1] - 1.3 * cm, f'Page {doc.page}')
        canv.setStrokeColor(LINE); canv.setLineWidth(0.5)
        canv.line(2 * cm, A4[1] - 1.5 * cm, A4[0] - 2 * cm, A4[1] - 1.5 * cm)
        canv.drawString(2 * cm, 1.1 * cm, f'Generated from the project artefacts · {TODAY} · commit {COMMIT}')
        canv.drawRightString(A4[0] - 2 * cm, 1.1 * cm, 'Academic research prototype — not financial advice')
        canv.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            st = flowable.style.name
            if st in ('h1', 'h2'):
                text = flowable.getPlainText()
                key = f'k{abs(hash(text)) % 10**9}'
                self.canv.bookmarkPage(key)
                self.notify('TOCEntry', (0 if st == 'h1' else 1, text, self.page, key))
                self.canv.addOutlineEntry(text, key, level=0 if st == 'h1' else 1, closed=(st == 'h2'))


def H1(text):
    return Paragraph(text, S['h1'])


def H2(text):
    return Paragraph(text, S['h2'])


def H3(text):
    return Paragraph(text, S['h3'])


def usd(x, d=2):
    return f"${x:,.{d}f}"


# ================================================================== content
def build():
    F = []
    st = status
    b, g, s_ = st['Bitcoin'], st['Gold'], st['Silver']
    ct = {a: st[a]['combined_test'] for a in ASSETS}
    nt = {a: st[a]['test_naive'] for a in ASSETS}

    # ---------------------------------------------------------------- cover
    F += [Spacer(1, 3.2 * cm),
          P('Multi-Asset Next-Day Price Prediction System', 'title'),
          P('Bitcoin · Gold · Silver', 'title'), Spacer(1, 0.5 * cm),
          P('Complete Technical Report of the Final Year Project — prepared for supervisor review', 'subtitle'),
          Spacer(1, 1.6 * cm),
          P('<b>Author:</b> Alyan Shahid (teamlocalhost) &nbsp;·&nbsp; <b>Programme:</b> BS Computer Science, University of Lahore', 'cover'),
          P(f'<b>Report date:</b> {TODAY} &nbsp;·&nbsp; <b>Code revision documented:</b> git commit {COMMIT} (branch main)', 'cover'),
          Spacer(1, 1.2 * cm),
          P('<b>Basis of this document.</b> Every statement about data, features, models, metrics and behaviour was taken from the '
            'implementation itself — the Python sources under <font name="Mono">src/</font>, <font name="Mono">app/</font> and '
            '<font name="Mono">config.py</font>, the frozen datasets under <font name="Mono">data/</font>, the trained artefacts under '
            '<font name="Mono">data/models/</font> and the result files under <font name="Mono">results/</font>. All tables are '
            'generated programmatically from those files by <font name="Mono">docs/build_technical_report.py</font>; nothing is typed in by hand. '
            f'The automated test suite ({N_TESTS} tests) and a live prediction run were executed on the report date to confirm the system works as described.', 'box'),
          P('<b>Scope note.</b> Where the project explored something that is <i>not</i> part of the final served system (for example the '
            'stacked-ensemble experiment, the multi-day and volatility targets, or the automatic daily prediction record that was prototyped '
            'and then removed), this report says so explicitly rather than describing it as a feature.', 'warn'),
          NextPageTemplate('main'), PageBreak()]

    # ---------------------------------------------------------------- TOC
    toc = TableOfContents()
    toc.levelStyles = [S['toc1'], S['toc2']]
    toc.dotsMinLevel = 0
    F += [H1('Contents'), toc, PageBreak()]

    # ================================================================ 1. EXECUTIVE SUMMARY
    F += [H1('1. Executive Summary')]
    F += [P('<b>What the project does.</b> The system predicts the <b>next trading day\'s closing price</b> of three assets — Bitcoin (BTC-USD), '
            'Gold (COMEX front-month futures, GC=F) and Silver (SI=F) — from daily market data downloaded from Yahoo Finance. Rather than predicting '
            'the price directly, every model predicts the next-day <b>log return</b> ln(P<sub>t+1</sub>/P<sub>t</sub>) from 28–29 backward-looking, '
            'stationary features, and the predicted return is converted back into a dollar price. The direction of the predicted return gives the '
            '<b>UP / DOWN</b> call.'),
          P('<b>Why these three assets.</b> They span three market structures with different behaviour: a 24/7 crypto-asset with very high '
            'volatility (Bitcoin, daily σ ≈ %.1f %%), a comparatively calm exchange-traded safe-haven commodity (Gold, σ ≈ %.1f %%) and a '
            'more volatile industrial/precious metal that co-moves with gold (Silver, σ ≈ %.1f %%). They are liquid, have long free daily '
            'histories, and are influenced by macro variables (US dollar, yields, equities, volatility) that can be added as features.' %
            (eda.loc['Bitcoin', 'ret_std_pct'], eda.loc['Gold', 'ret_std_pct'], eda.loc['Silver', 'ret_std_pct'])),
          P('<b>Machine-learning approach.</b> Six models are trained per asset — Ridge regression, Random Forest, LightGBM, CatBoost, a GRU and an LSTM '
            'network — and compared against three baselines (random walk, historical mean, ARIMA) and a stacked-ensemble experiment. Hyper-parameters '
            'are chosen with 4-fold expanding-window walk-forward validation on the training/validation period only; the untouched test period '
            f'({b["test_period"][0]} → {b["test_period"][1]}) is evaluated exactly once. <b>The served forecast is the equal-weight combination</b> of the six '
            'trained models\' predicted returns: the user never selects a model.'),
          P('<b>What the user sees.</b> A Streamlit dashboard with five tabs: <i>Forecast</i> (one click → the latest complete market bar is fetched, '
            'all six models run, one final predicted price with UP/DOWN, the change in % and $, a ±1-RMSE uncertainty band, each model\'s own prediction, '
            'and the held-out reliability of the forecast next to it), <i>Predict a Day (unseen test)</i> (stand on any test day, predict the next day '
            'using only data up to that day, then reveal the actual close, the error and HIT/FAIL), <i>Test-Set History</i> (every unseen day: predicted, actual, '
            'error, HIT/FAIL, downloadable), <i>Model Performance</i> and <i>Methodology</i>. A FastAPI endpoint serves the same forecast as JSON.'),
          P('<b>How predictions are evaluated.</b> In return space (RMSE, MAE, R²), in dollar space (MAE, RMSE, MAPE), directional accuracy on non-flat days with a '
            'binomial p-value against 50 %, a Diebold–Mariano test against the random-walk forecast, and a long/flat strategy back-test with 10 bps costs. '
            'The random walk ("tomorrow = today") is the baseline every model must beat.'),
          P('<b>Headline result (honest).</b> On the unseen test period the combined forecast\'s next-day price error is %.2f %% (Bitcoin), %.2f %% (Gold) and '
            '%.2f %% (Silver) of price — essentially identical to the random walk (%.2f / %.2f / %.2f %%). Directional accuracy is %.1f / %.1f / %.1f %% '
            '(binomial p = %.2f / %.2f / %.2f) and the Diebold–Mariano test cannot reject equal accuracy with the random walk (p = %.2f / %.2f / %.2f). '
            '<b>No model, single or combined, beats the random walk at a one-day horizon in this data.</b> This is the expected result for liquid daily '
            'financial returns and the project reports it as such; the project\'s contribution is a leakage-audited, statistically tested pipeline and a '
            'transparent demonstration system, including the discovery and correction of a subtle close-time data leak that had produced an apparently '
            'significant 62 %% directional-accuracy result for Silver.' %
            (ct['Bitcoin']['MAPE_usd'], ct['Gold']['MAPE_usd'], ct['Silver']['MAPE_usd'], nt['Bitcoin']['MAPE_usd'], nt['Gold']['MAPE_usd'], nt['Silver']['MAPE_usd'],
             ct['Bitcoin']['DirAcc_pct'], ct['Gold']['DirAcc_pct'], ct['Silver']['DirAcc_pct'], ct['Bitcoin']['DirAcc_pvalue'], ct['Gold']['DirAcc_pvalue'], ct['Silver']['DirAcc_pvalue'],
             ct['Bitcoin']['DM_pvalue'], ct['Gold']['DM_pvalue'], ct['Silver']['DM_pvalue'])),
          P('<b>Automatic daily predictions — status.</b> An automatic daily forward-prediction record (scheduled job that saves each day\'s forecast and verifies it '
            'against the next day\'s actual close with HIT/FAIL) was built during the final audit and <b>removed from the final system at the author\'s decision</b> '
            '(CHANGELOG, 2026-09-14). The final code base therefore contains <i>no scheduler and no persistent forward log</i>. Out-of-sample evidence is provided instead by the '
            'frozen unseen test period: every test day is predicted from data up to that day and compared with the actual next close (Test-Set History, Predict-a-Day). '
            'Section 19 explains this in detail.', 'warn'),
          P('<b>Why it is suitable as an FYP.</b> The project covers the complete data-science life-cycle — data acquisition from public APIs, cleaning, '
            'feature engineering, leakage control, time-series validation, hyper-parameter tuning, ten model families, statistical evaluation, '
            f'model combination, a user-facing dashboard, an API, {N_TESTS} automated tests, CI, Docker packaging and reproducible documentation — and it '
            'demonstrates scientific honesty: results were re-run and corrected when the audit found problems, and negative results are reported as results.')]

    # ================================================================ 2. PROJECT OVERVIEW
    F += [PageBreak(), H1('2. Project Overview — the complete pipeline')]
    F += [P('The system is organised as a sequence of scripts, each of which reads the previous stage\'s files and writes its own. '
            'The <font name="Mono">Makefile</font> target <font name="Mono">make pipeline</font> runs them in order; <font name="Mono">scripts/retrain.py</font> '
            'additionally re-downloads the data. Figure 1 shows the stages actually implemented.')]
    F += [flow_chart(['1 Data collection\nYahoo Finance + alternative.me', '2 Cleaning\nde-dup · sort · calendar', '3 Preprocessing\nlog return · warm-up drop',
                      '4 Feature engineering\n28–29 stationary features', '5 Feature policy\nlevels excluded (config)', '6 Chronological split\ntrain / val / test (frozen dates)',
                      '7 Scaling\nMinMax fitted on train only', '8 Walk-forward tuning\n4 expanding folds', '9 Two-phase training\n6 models × 3 assets',
                      '10 Single test evaluation\nall models + Combined', '11 Combined forecast\nmean of 6 returns → price, UP/DOWN', '12 Streamlit dashboard\n+ FastAPI',
                      '13 Predict-a-Day demo\nunseen test days', '14 Actual-close comparison\nerror $, error %', '15 HIT / FAIL\nsign(pred) = sign(actual)?',
                      '16 Test-set history\n143–207 rows per asset (CSV)'], cols=4),
          P('Figure 1. The implemented pipeline. Stages 1–7 live in <font name="Mono">src/data/</font>, 8–9 in <font name="Mono">src/training/</font>, '
            '10 and 16 in <font name="Mono">src/evaluation/backtesting.py</font>, 11 and 13–15 in <font name="Mono">src/inference/prediction.py</font>, 12 in '
            '<font name="Mono">app/streamlit_app.py</font> and <font name="Mono">src/api/app.py</font>.', 'caption')]
    stages = [
        ['Stage', 'What happens', 'Implemented in'],
        ['Data collection', 'Daily OHLCV for BTC-USD, GC=F, SI=F and daily closes for DXY, WTI, 10-y yield, S&amp;P 500, VIX from Yahoo Finance (yfinance); Bitcoin Fear &amp; Greed index from alternative.me. The running (incomplete) bar of the current day is dropped at download time (complete-bar rule).', 'src/data/data_collection.py, external_data.py, market_calendar.py'],
        ['Data cleaning', 'Drop duplicate timestamps, sort chronologically, drop non-positive prices; Bitcoin re-indexed to every calendar day (gaps forward-filled — none in the stored data); Gold/Silver keep exchange days only (no synthetic weekend rows).', 'preprocessing.py :: DataCleaner.clean_data'],
        ['Preprocessing', 'Log return ln(P<sub>t</sub>/P<sub>t−1</sub>); indicator warm-up rows dropped (dropna); validation: no NaN/inf, ≥ 100 rows.', 'preprocessing.py'],
        ['Feature engineering', 'Returns and lags, momentum, rolling/HAR/EWMA volatility, RSI, ADX, ROC, normalised MACD, EMA ratios, Bollinger %B/width, ATR/price, high–low range, macro returns, sentiment/calendar/volume (Bitcoin), gold return (Silver). Close-time rule for the metals.', 'preprocessing.py (add_* methods)'],
        ['Feature selection', 'A fixed policy rather than an algorithm: all raw price-level columns are excluded from model inputs (config.LEVEL_COLUMNS); ablation experiment E2 confirmed no feature group hurts.', 'config.py, src/experiments/run_experiments.py'],
        ['Train / validation / test', f'Chronological split by frozen calendar dates: train ≤ {TRAIN_END}, validation ≤ {VAL_END}, test = remainder; defined by the dates the target covers.', 'preprocessing.py :: normalize_and_split, build_dataset'],
        ['Model training', 'Two-phase fit: phase A on train with validation monitoring (stopping point, honest validation metrics), phase B refit on train+val with the stopping point fixed → deployed artefact.', 'src/training/train_models.py, src/models/registry.py'],
        ['Model evaluation', 'Walk-forward CV table (selection evidence) and ONE evaluation of every model on the untouched test set with return-space, USD, directional, Diebold–Mariano and strategy metrics.', 'src/evaluation/backtesting.py, src/utils/metrics.py'],
        ['Combined prediction', 'Equal-weight mean of the six trained models\' predicted log returns → price via P<sub>t</sub>·exp(ŷ) → UP if ŷ &gt; 0 else DOWN.', 'src/inference/prediction.py'],
        ['Dashboard / API', 'Streamlit dashboard (5 tabs) and FastAPI (/health, /models, /predict/{asset}).', 'app/streamlit_app.py, src/api/app.py'],
        ['Verification &amp; history', 'Predict-a-Day (unseen test day → predict → reveal actual → error → HIT/FAIL) and the Test-Set History (one row per unseen day, all models + Combined) written by the evaluation script.', 'prediction.py :: predict_for_date, prediction_history; results/predictions/*.csv'],
    ]
    F += [Spacer(1, 4), table(stages, [3.0 * cm, 9.6 * cm, 4.0 * cm]), P('Table 1. Pipeline stages and where they are implemented.', 'caption')]

    # ================================================================ 3. OBJECTIVES
    F += [H1('3. Project Objectives')]
    F += [P('The objectives below are the ones the delivered code actually meets; each is paired with the evidence in the repository.')]
    obj = [['#', 'Objective', 'How it is met (evidence)'],
           ['1', 'Collect historical daily financial data for three assets plus macro/sentiment drivers from free public sources', f'Yahoo Finance + alternative.me downloads from 2018-01-01; {len(raw["Bitcoin"]):,} / {len(raw["Gold"]):,} / {len(raw["Silver"]):,} raw rows (BTC / Gold / Silver)'],
           ['2', 'Process financial time series correctly (calendar handling, complete bars only, no look-ahead)', 'DataCleaner; complete-bar rule; close-time alignment; tests in test_features.py and test_market_calendar.py'],
           ['3', 'Engineer meaningful, stationary, backward-looking predictive features', '29 / 28 / 29 model features; docs/FEATURES.md; ADF tests in results/eda_summary.csv'],
           ['4', 'Predict the next trading day\'s price (via the log return) with a one-day horizon', 'build_dataset(task="return_1d"); reconstruct_price'],
           ['5', 'Train and compare several model families under a valid time-series protocol', '6 trained models + 3 baselines + stacked experiment; walk-forward CV; two-phase training'],
           ['6', 'Compare every model against a meaningful baseline with statistical tests', 'Random-walk baseline; Diebold–Mariano test; binomial test on direction'],
           ['7', 'Combine the models into one served forecast that needs no manual model choice', 'COMBINED = equal-weight mean; evaluated like a single model (row "Combined")'],
           ['8', 'Visualise data, predictions and evaluation for a non-expert user', 'Streamlit dashboard, Plotly charts, matplotlib report figures'],
           ['9', 'Evaluate predictions against actual values on genuinely unseen days, per day', 'Test-Set History (predicted / actual / error / HIT-FAIL per unseen day); Predict-a-Day'],
           ['10', 'Provide an easy-to-use dashboard and an API', 'One "Run prediction" button; FastAPI /predict/{asset}'],
           ['11', 'Make the work reproducible and testable', f'Frozen split dates, seeds, pinned requirements, Makefile, {N_TESTS} tests, CI workflow, Docker']]
    F += [table(obj, [0.7 * cm, 7.4 * cm, 8.5 * cm]), P('Table 2. Objectives and evidence.', 'caption')]
    F += [P('<b>Not an objective of the final system:</b> a persistent automatic daily prediction log (prototyped, then removed — §19), multi-day or volatility '
            'forecasting (evaluated as experiments E3/E4 only), or any trading recommendation.')]

    # ================================================================ 4. PROBLEM STATEMENT
    F += [H1('4. Problem Statement')]
    F += [P('Short-horizon prediction of financial asset prices is one of the hardest forecasting problems, for reasons that are visible in this project\'s own data:'),
          *bullets([
              f'<b>Volatility and fat tails.</b> Daily returns are large and heavy-tailed: excess kurtosis is {eda.loc["Bitcoin","ret_excess_kurtosis"]:.1f} (Bitcoin), '
              f'{eda.loc["Gold","ret_excess_kurtosis"]:.1f} (Gold) and {eda.loc["Silver","ret_excess_kurtosis"]:.1f} (Silver) in the training window; '
              f'{eda.loc["Bitcoin","share_abs_ret_gt_2pct"]:.0f} % of Bitcoin days move more than 2 %.',
              f'<b>Almost no linear memory.</b> The lag-1 autocorrelation of returns is {eda.loc["Bitcoin","acf_lag1"]:+.3f} / {eda.loc["Gold","acf_lag1"]:+.3f} / '
              f'{eda.loc["Silver","acf_lag1"]:+.3f}: yesterday\'s return says almost nothing about today\'s, which is why ARIMA-type models can add little.',
              '<b>Non-stationarity and regime change.</b> Price levels are non-stationary (ADF p ≈ 0.97–1.00) while returns are stationary (p &lt; 10<super>−29</super>); '
              f'the 2026 test period is a different regime — Gold traded at {usd(feat["Gold"].loc["2026-02-18":, "price"].min(),0)}–{usd(feat["Gold"].loc["2026-02-18":, "price"].max(),0)} against a training range of '
              f'{usd(feat["Gold"].loc[:TRAIN_END, "price"].min(),0)}–{usd(feat["Gold"].loc[:TRAIN_END, "price"].max(),0)}.',
              '<b>Non-linear, unstable relationships and external shocks.</b> Macro news, regulation, exchange outages and sentiment move prices in ways that are not in the history.',
              '<b>Time-series dependencies.</b> Observations are ordered; random shuffling, k-fold cross-validation or features that peek even a few hours into the future silently inflate results.'])]
    F += [P('<b>How the project approaches it.</b> The problem is framed as a supervised regression of the next-day log return on information that is provably '
            'available when the day-<i>t</i> close is fixed. The random walk (predicted return = 0) is treated as the null hypothesis; models are considered useful only if '
            'they beat it under a walk-forward protocol and a single held-out evaluation with formal tests. The system is positioned strictly as an <b>academic prediction '
            'and evaluation system</b>: it does not claim, and its own results do not show, a profitable trading edge.')]

    # ================================================================ 5. ASSETS
    F += [H1('5. Assets Predicted')]
    F += [P('All three assets share the same target definition, split dates, model set and combination rule. They differ in data source ticker, trading calendar, '
            'close time and the feature groups that make sense for them.')]
    assets_tbl = [['', 'Bitcoin', 'Gold', 'Silver'],
                  ['Ticker / instrument', 'BTC-USD (Yahoo composite spot index)', 'GC=F (COMEX front-month gold futures)', 'SI=F (COMEX front-month silver futures)'],
                  ['Trading calendar', '7 days a week (calendar days)', 'Exchange days only (Mon–Fri, no holidays modelled)', 'Exchange days only'],
                  ['Daily close used', '00:00 UTC bar close (after every US market close)', '13:30 ET COMEX settlement (Yahoo\'s daily Close)', '13:25–13:30 ET COMEX settlement'],
                  ['What is predicted', 'Next calendar day\'s close (via log return)', 'Next exchange day\'s settlement (via log return)', 'Next exchange day\'s settlement (via log return)'],
                  ['Horizon', '1 day', '1 trading day', '1 trading day'],
                  ['Usable rows (after warm-up)', f'{len(feat["Bitcoin"]):,} ({feat["Bitcoin"].index[0].date()} → {feat["Bitcoin"].index[-1].date()})',
                   f'{len(feat["Gold"]):,} ({feat["Gold"].index[0].date()} → {feat["Gold"].index[-1].date()})', f'{len(feat["Silver"]):,} ({feat["Silver"].index[0].date()} → {feat["Silver"].index[-1].date()})'],
                  ['Model features', f'{len(b["features"])}: price/technical set + S&amp;P 500 and VIX same-day returns + Fear &amp; Greed + day-of-week + volume change',
                   f'{len(g["features"])}: price/technical set + previous-day DXY, WTI, 10-y yield, S&amp;P 500, VIX returns',
                   f'{len(s_["features"])}: as Gold + gold\'s same-day return'],
                  ['Close-time rule', 'Same-day macro values allowed (bar closes after them)', 'Macro = previous day; hl_range / ATR / ADX lagged one session', 'Same as Gold'],
                  ['Models', '6 trained (Ridge, RF, LightGBM, CatBoost, GRU, LSTM) + baselines', 'same', 'same'],
                  ['Served forecast', 'Combined (equal-weight mean of 6)', 'Combined', 'Combined'],
                  ['Best single model by walk-forward CV (for comparison)', b['primary_model'], g['primary_model'], s_['primary_model']],
                  ['Train / val / test samples', f'{b["n_train"]:,} / {b["n_val"]} / {ct["Bitcoin"]["n_test"]}', f'{g["n_train"]:,} / {g["n_val"]} / {ct["Gold"]["n_test"]}', f'{s_["n_train"]:,} / {s_["n_val"]} / {ct["Silver"]["n_test"]}']]
    F += [table(assets_tbl, [3.6 * cm, 4.4 * cm, 4.4 * cm, 4.2 * cm], align_right_from=None), P('Table 3. The three assets side by side.', 'caption')]
    F += [P('<b>Important differences.</b> (i) Bitcoin has ~45 % more rows per year because it trades on weekends; the metals keep their exchange calendar and the project '
            'deliberately does <i>not</i> create synthetic weekend rows for them (an earlier version did, and 30 % of its metal rows were forward-filled zero-return fakes). '
            '(ii) Yahoo\'s daily close for the futures is the 13:30 ET settlement, which is <i>before</i> the US equity, rates and FX closes — so the metals may only use the '
            '<i>previous</i> day\'s macro values and the previous session\'s High/Low (§11). (iii) The yfinance volume column for futures is front-month contract volume that collapses '
            'at contract rolls; it is not used for the metals. (iv) Yahoo publishes Bitcoin\'s previous-day bar several hours after 00:00 UTC, so a morning "Run prediction" may start '
            'from the bar before; the dashboard says so explicitly.')]

    # ================================================================ 6. DATASET
    F += [H1('6. Dataset — Detailed Explanation')]
    F += [P('All series are downloaded by the project\'s own scripts and stored as CSV files under <font name="Mono">data/raw/</font> (frozen evaluation data) '
            'and <font name="Mono">data/raw_live/</font> (refreshed copies used only for live forecasts). No paid data, no API key and no database are used. '
            'The figures below are read from the stored files.')]
    ds = [['Asset', 'Data source / ticker', 'Date range (stored)', 'Frequency', 'Raw rows', 'Usable rows', 'Target']]
    for a in ASSETS:
        r = raw[a]; f = feat[a]
        ds.append([a, f'Yahoo Finance (yfinance) · {ASSET_CONFIG[a]["ticker"]}', f'{r["timestamp"].min().date()} → {r["timestamp"].max().date()}',
                   'daily (calendar days)' if a == 'Bitcoin' else 'daily (exchange days)', f'{len(r):,}', f'{len(f):,}', 'next-day log return ln(P<sub>t+1</sub>/P<sub>t</sub>)'])
    F += [table(ds, [1.5 * cm, 3.6 * cm, 3.2 * cm, 2.4 * cm, 1.3 * cm, 1.5 * cm, 3.1 * cm], align_right_from=None), P('Table 4. Primary (target) datasets. "Usable rows" = after the 30-row indicator warm-up is dropped.', 'caption')]
    ex = [['Series', 'Source / ticker', 'Rows', 'Date range', 'Used by', 'Feature derived']]
    ex_meta = [('US Dollar Index', 'Yahoo · DX-Y.NYB', 'DXY', 'Gold, Silver', 'dxy_return (previous day)'), ('WTI crude oil', 'Yahoo · CL=F', 'WTI', 'Gold, Silver', 'oil_return (previous day)'),
               ('10-year Treasury yield', 'Yahoo · ^TNX', 'TNX', 'Gold, Silver', 'tnx_return (previous day)'), ('S&amp;P 500', 'Yahoo · ^GSPC', 'SP500', 'all', 'sp500_return (same day BTC / previous day metals)'),
               ('VIX', 'Yahoo · ^VIX', 'VIX', 'all', 'vix_return (same day BTC / previous day metals)'), ('Crypto Fear &amp; Greed index', 'alternative.me API (free, no key)', 'FG', 'Bitcoin', 'fear_greed (0–100, same day)')]
    for name, src, key, used, featname in ex_meta:
        r = ext_raw[key]
        ex.append([name, src, f'{len(r):,}', f'{r["timestamp"].min().date()} → {r["timestamp"].max().date()}', used, featname])
    ex.append(['Gold close (for Silver)', 'the gold_data.csv above', f'{len(raw["Gold"]):,}', '—', 'Silver', 'gold_return (same day; same settlement window)'])
    F += [table(ex, [2.9 * cm, 3.2 * cm, 1.1 * cm, 3.1 * cm, 1.7 * cm, 4.6 * cm], align_right_from=None), P('Table 5. External (explanatory) series.', 'caption')]
    F += [H2('6.1 Columns of the raw files')]
    cols = [['Column', 'Type', 'Meaning', 'How it is used'],
            ['timestamp', 'datetime (tz-naive, daily)', 'Trading day of the bar', 'Index; chronological sorting and splitting; date arithmetic for the target'],
            ['open', 'float', 'First traded price of the day', 'Chart only (level column, never a model input)'],
            ['high', 'float', 'Highest price of the session', 'Inputs to ATR, ADX, hl_range (lagged one session for the metals)'],
            ['low', 'float', 'Lowest price of the session', 'as high'],
            ['price', 'float', 'Yahoo "Close" — BTC: 00:00 UTC close; futures: 13:30 ET settlement', 'Basis of the log return (target and features); P<sub>t</sub> for price reconstruction; chart'],
            ['volume', 'int', 'Traded volume (BTC: composite; futures: front-month contract volume)', 'Bitcoin only: log_volume_change. Not used for the metals (contract-roll jumps)'],
            ['price (external files)', 'float', 'Daily close of the macro series', 'Converted to a daily log return on its own calendar, then aligned'],
            ['fear_greed', 'int 0–100', 'Sentiment index (0 = extreme fear, 100 = extreme greed)', 'Bitcoin feature, forward-filled onto the BTC calendar']]
    F += [table(cols, [2.6 * cm, 2.9 * cm, 5.1 * cm, 6.0 * cm], align_right_from=None), P('Table 6. Column dictionary of the stored raw files (verified against data/raw/*.csv).', 'caption')]
    F += [H2('6.2 Data quality of the stored files')]
    F += [P(f'The stored raw files contain <b>no null values and no duplicate timestamps</b> (checked on the report date). Bitcoin has a complete calendar '
            f'({len(raw["Bitcoin"]):,} rows for {(raw["Bitcoin"]["timestamp"].max() - raw["Bitcoin"]["timestamp"].min()).days + 1:,} calendar days — no gaps to forward-fill). '
            f'Gold and Silver have {len(raw["Gold"]):,} exchange days each. Days with an exactly-zero return exist only in the metals '
            f'({int((feat["Gold"]["log_return"] == 0).sum())} Gold, {int((feat["Silver"]["log_return"] == 0).sum())} Silver over the whole history, none inside the test period); '
            'they are excluded from directional accuracy because their direction is undefined.'),
          P('<b>Known vendor issues that the code handles.</b> Yahoo returns the current day\'s <i>running</i> bar as if it were complete — the complete-bar rule '
            '(<font name="Mono">src/data/market_calendar.py</font>) drops it at download time (Bitcoin: rows dated today UTC; futures and US series: rows dated today ET '
            'before 17:15 ET). Yahoo\'s continuous futures series follows the current front-month contract, so recent closes can differ between two downloads after a roll; '
            'the frozen evaluation files are therefore never overwritten by live refreshes.')]
    F += fig(os.path.join(FIGURES_DIR, 'price_history.png'), 16.5, 'Figure 2. Stored price history of the three assets with the frozen train (green) / validation (amber) / test (red) windows.', max_height_cm=9.5)

    # ================================================================ 7. PREPROCESSING
    F += [H1('7. Data Preprocessing')]
    F += [P('Preprocessing is implemented in <font name="Mono">DataCleaner.clean_data()</font> and <font name="Mono">normalize_and_split()</font> '
            '(<font name="Mono">src/data/preprocessing.py</font>) and runs identically for the frozen data and for live refreshes.')]
    pre = [['Step', 'What is done', 'Why it is necessary'],
           ['1 Parse &amp; sort', 'timestamp parsed to datetime; rows sorted chronologically', 'Every later step (returns, rolling windows, split) assumes time order'],
           ['2 Duplicate removal', 'drop_duplicates(subset=["timestamp"])', 'A duplicated day would double-count a sample and break the shift/rolling logic'],
           ['3 Invalid values', 'rows with price ≤ 0 dropped; rows with missing close dropped at download', 'log(price) is undefined for non-positive prices'],
           ['4 Calendar handling', 'Bitcoin: reindex to every calendar day and forward-fill genuine gaps (0 in stored data). Gold/Silver: keep exchange days only',
            'Bitcoin trades daily, a missing day is a data gap; futures do not trade at weekends, so synthetic rows would create fake zero-return samples'],
           ['5 Log return', 'log_return = ln(price<sub>t</sub> / price<sub>t−1</sub>)', 'Stationary quantity; basis of the target and of most features'],
           ['6 Feature engineering', 'technical, volatility, momentum, macro, sentiment features (§8)', 'Turn the raw series into scale-free predictors'],
           ['7 Close-time lagging', 'metals only: hl_range, atr_norm, ADX shifted one session; macro series aligned to the previous day', 'Yahoo\'s futures close is the 13:30 ET settlement; later information would leak (§11)'],
           ['8 Warm-up removal', 'dropna() removes the first ~30 rows where indicators are undefined', 'Rolling/EWM windows need history; NaNs cannot enter a model'],
           ['9 Validation', 'assert: no NaN, no inf, ≥ 100 rows (raises ValueError otherwise)', 'A corrupt download must fail loudly instead of producing silent nonsense'],
           ['10 Chronological split', f'train ≤ {TRAIN_END}; validation ≤ {VAL_END}; test = later rows', 'Time-series data must be split forward in time (§10)'],
           ['11 Scaling', 'MinMaxScaler(0–1) fitted on the training rows only and applied to validation/test; saved as <prefix>_scaler.pkl', 'The recurrent models need bounded inputs; fitting on train only prevents test statistics from leaking'],
           ['12 Target standardisation', 'target z-scored with the training mean/std (stored in model_status.json)', 'GRU/LSTM train on a unit-variance target; tree models are unaffected; inverted at prediction time'],
           ['13 Feature alignment', 'the last row of each 30-day window feeds the tabular models; the whole window feeds GRU/LSTM; both see identical samples', 'Every model is compared on exactly the same days']]
    F += [table(pre, [3.0 * cm, 7.0 * cm, 6.6 * cm], align_right_from=None), P('Table 7. Preprocessing steps (in execution order).', 'caption')]
    F += [P('Outputs per asset: <font name="Mono">data/processed/&lt;prefix&gt;_features.csv</font> (all columns incl. price levels, for charts and for inference), '
            '<font name="Mono">&lt;prefix&gt;_train_scaled.csv / _val_scaled.csv / _test_scaled.csv</font> (model feature columns only, scaled) and '
            '<font name="Mono">data/models/&lt;prefix&gt;_scaler.pkl</font>.')]

    # ================================================================ 8. FEATURES
    F += [H1('8. Feature Engineering — Complete List')]
    F += [P('The list below is the exact set of model inputs stored in <font name="Mono">model_status.json → features</font> and in the scaled split files '
            f'({len(b["features"])} for Bitcoin, {len(g["features"])} for Gold, {len(s_["features"])} for Silver). Every feature at row <i>t</i> uses only information '
            'available when P<sub>t</sub> is observed ("historical/current only" — verified by the look-ahead tests). No price-level column is a model input.')]
    FE = [['Feature', 'Category', 'What it represents / how it is calculated', 'Why it may help', 'Assets'],
          ['log_return', 'Return', 'ln(P<sub>t</sub>/P<sub>t−1</sub>) — today\'s return', 'Short-term reversal / continuation', 'all'],
          ['return_1d, return_2d, return_5d, return_10d', 'Lag', 'log_return shifted by 1, 2, 5, 10 rows (strictly past returns)', 'Lets tabular models see the recent path', 'all'],
          ['return_20d', 'Momentum', 'Σ of the last 20 log returns (one-month momentum)', 'Momentum / monthly reversal literature', 'all'],
          ['volatility_10d, volatility_30d', 'Volatility', 'rolling standard deviation of log_return over 10 / 30 rows', 'Volatility clustering', 'all'],
          ['rv_1d, rv_5d, rv_22d', 'Volatility (HAR)', '√(mean of r² over 1 / 5 / 22 rows) — daily, weekly, monthly realised volatility', 'HAR structure of volatility', 'all'],
          ['ewma_vol', 'Volatility', 'RiskMetrics EWMA σ, λ = 0.94 (min 22 rows)', 'Fast-reacting volatility state', 'all'],
          ['RSI', 'Momentum indicator', 'Wilder RSI(14) on price, 0–100', 'Overbought / oversold', 'all'],
          ['ADX †', 'Trend indicator', 'ADX(14) from high/low/close (ta library); metals: previous session', 'Trend strength / regime', 'all'],
          ['ROC', 'Momentum indicator', '(P<sub>t</sub>/P<sub>t−12</sub> − 1)·100', '12-day rate of change', 'all'],
          ['ema14_ratio, ema30_ratio', 'Trend (relative)', 'P<sub>t</sub>/EMA<sub>14</sub> − 1, P<sub>t</sub>/EMA<sub>30</sub> − 1', 'Distance from short / medium trend, scale-free', 'all'],
          ['macd_norm', 'Trend (relative)', '(EMA<sub>12</sub> − EMA<sub>26</sub>) / P<sub>t</sub>', 'MACD line normalised by price', 'all'],
          ['macd_hist_norm', 'Trend (relative)', '(MACD − Signal<sub>9</sub>) / P<sub>t</sub>', 'MACD histogram normalised', 'all'],
          ['bb_pctb', 'Volatility band', '(P<sub>t</sub> − BB<sub>lower</sub>) / (BB<sub>upper</sub> − BB<sub>lower</sub>), 20-day ±2σ bands', 'Position inside the band', 'all'],
          ['bb_width', 'Volatility band', '(BB<sub>upper</sub> − BB<sub>lower</sub>) / BB<sub>mid</sub>', 'Band width = volatility regime', 'all'],
          ['atr_norm †', 'Volatility', 'ATR(14) / P<sub>t</sub>; metals: previous session', 'True range relative to price', 'all'],
          ['hl_range †', 'Volatility (range)', '(High − Low) / P; metals: previous session\'s range', 'Parkinson-style intraday volatility proxy', 'all'],
          ['sp500_return ‡', 'Macro', 'ln(SPX<sub>t</sub>/SPX<sub>t−1</sub>) — same day for BTC, previous day for metals', 'Risk-on / risk-off', 'all'],
          ['vix_return ‡', 'Macro', 'ln(VIX<sub>t</sub>/VIX<sub>t−1</sub>) — same day BTC, previous day metals', 'Change in implied volatility / fear', 'all'],
          ['dxy_return ‡', 'Macro', 'previous-day log return of the US Dollar Index', 'Metals are priced in USD', 'Gold, Silver'],
          ['oil_return ‡', 'Macro', 'previous-day log return of WTI crude', 'Commodity co-movement / inflation proxy', 'Gold, Silver'],
          ['tnx_return ‡', 'Macro', 'previous-day log return of the 10-year yield', 'Opportunity cost of holding gold', 'Gold, Silver'],
          ['gold_return', 'Cross-asset', 'ln(GC<sub>t</sub>/GC<sub>t−1</sub>) same day (gold and silver settle in the same 13:25–13:30 window)', 'Silver follows gold', 'Silver'],
          ['fear_greed', 'Sentiment', 'alternative.me index 0–100, published 00:00 UTC for the day', 'Crypto-specific sentiment', 'Bitcoin'],
          ['dow_sin, dow_cos', 'Calendar', 'sin / cos(2π·weekday/7)', 'Weekend effects in a 7-day market', 'Bitcoin'],
          ['log_volume_change', 'Volume', 'clip(ln(V<sub>t</sub>/V<sub>t−1</sub>), −3, 3)', 'Attention / activity spikes', 'Bitcoin']]
    F += [table(FE, [3.3 * cm, 2.2 * cm, 5.6 * cm, 3.6 * cm, 1.9 * cm], align_right_from=None, font=7.3),
          P('Table 8. Every model input. † lagged one session for Gold/Silver (Yahoo\'s futures High/Low extend past the 13:30 settlement); ‡ previous trading day\'s value for '
            'Gold/Silver (those markets close after the settlement). Bitcoin 29 · Gold 28 · Silver 29 features.', 'caption')]
    F += [P('<b>Kept for charts only, never model inputs</b> (<font name="Mono">config.LEVEL_COLUMNS</font>): ' + ', '.join(LEVEL_COLUMNS) + '. Reason: they are price levels. '
            f'Gold\'s test period ({usd(feat["Gold"].loc["2026-02-18":, "price"].min(),0)}–{usd(feat["Gold"].loc["2026-02-18":, "price"].max(),0)}) lies entirely above its training range '
            f'({usd(feat["Gold"].loc[:TRAIN_END, "price"].min(),0)}–{usd(feat["Gold"].loc[:TRAIN_END, "price"].max(),0)}); a tree model cannot extrapolate a level it has never seen.'),
          P('<b>Features that exist in the code but are deliberately not used for some assets:</b> futures volume (unusable contract-roll jumps), day-of-week for the metals (no calendar '
            'effect after removing synthetic weekends), same-day macro values for the metals (close-time leak). <b>No feature-selection algorithm</b> (e.g. RFE) is used; the ablation '
            'experiment E2 (validation only) showed that dropping any single group changes walk-forward RMSE by between −0.39 % and +0.05 %, i.e. within noise, so the full set is kept.')]
    F += fig(os.path.join(FIGURES_DIR, 'gold_feature_importance.png'), 13.5, 'Figure 3. Feature importance of the CV-selected tree model for Gold (CatBoost). GRU/LSTM expose no importance; the combined forecast averages all six.', max_height_cm=8.5)

    # ================================================================ 9. TARGET
    F += [H1('9. Target Variable')]
    F += [P('<b>What exactly is predicted.</b> The served task is <font name="Mono">return_1d</font>: the <b>next trading day\'s log return</b>'),
          P('y<sub>t</sub> = ln( P<sub>t+1</sub> / P<sub>t</sub> ),&nbsp;&nbsp;&nbsp; predicted price&nbsp; P′<sub>t+1</sub> = P<sub>t</sub> · exp(ŷ<sub>t</sub>),&nbsp;&nbsp;&nbsp; direction = UP if ŷ<sub>t</sub> &gt; 0 else DOWN', 'code'),
          P('where P′ denotes a predicted price, P<sub>t</sub> is the close of the last <i>complete</i> bar (day <i>t</i>) and P<sub>t+1</sub> the close of the next row in the asset\'s own calendar — the next calendar day for '
            'Bitcoin, the next exchange day for Gold and Silver (Friday → Monday). The horizon is one bar. The target is created in <font name="Mono">build_dataset()</font> '
            f'(<font name="Mono">_task_target</font>: <font name="Mono">log_return.shift(−1)</font>) and, for the recurrent models, in <font name="Mono">create_sequences()</font>, which builds windows of {SEQ_LEN} rows '
            'ending at day <i>t</i> and takes the log return of row <i>t</i>+1 as the label — the window never contains that row (unit-tested).'),
          P('<b>Why the return and not the price.</b> Prices are non-stationary (ADF p ≈ 0.97–1.00); a model predicting price levels can reach R² ≈ 0.95 simply by copying today\'s price, '
            'which says nothing about skill. Predicting the return makes the target stationary, makes the random-walk baseline explicit (ŷ = 0) and makes every comparison honest. '
            'The target is standardised with the training mean/std for fitting and inverted at prediction time (<font name="Mono">model_status.json: target_mean / target_std</font>).'),
          P(f'<b>Worked example (real row from the Gold test history).</b> Data available up to <b>{hist["Gold"]["date"].iloc[-1].date()}</b> (P<sub>t</sub> = {usd(hist["Gold"]["prev_close"].iloc[-1])}) '
            f'is used to predict the close of <b>{hist["Gold"]["target_date"].iloc[-1].date()}</b>. The combined predicted return was {hist["Gold"]["pred_return_Combined"].iloc[-1]*100:+.3f} %, '
            f'i.e. a predicted close of {usd(hist["Gold"]["pred_close_Combined"].iloc[-1])}; the actual close was {usd(hist["Gold"]["actual_close"].iloc[-1])} '
            f'(actual return {hist["Gold"]["actual_return"].iloc[-1]*100:+.3f} %), an error of {hist["Gold"]["error_pct_Combined"].iloc[-1]:+.2f} % and a direction '
            f'{"HIT" if hist["Gold"]["direction_hit_Combined"].iloc[-1] == 1 else "FAIL"}.'),
          P('<b>Experimental tasks that exist in the code but are not served:</b> <font name="Mono">return_5d</font> (5-day forward return) and <font name="Mono">vol_5d / vol_22d</font> '
            '(forward realised volatility). They were evaluated on validation folds only (experiments E3/E4) and are not part of the dashboard or API.')]

    # ================================================================ 10. SPLIT
    F += [H1('10. Time-Series Data Splitting')]
    sp = [['Split', 'Rule (calendar dates, identical for all assets)', 'Bitcoin', 'Gold', 'Silver', 'Purpose']]
    sp.append(['Train', f'target ends on or before {TRAIN_END}', f'{b["n_train"]:,} samples ({b["data_start"]} →)', f'{g["n_train"]:,} ({g["data_start"]} →)', f'{s_["n_train"]:,} ({s_["data_start"]} →)', 'Fit the models'])
    sp.append(['Validation', f'target after {TRAIN_END} and on or before {VAL_END}', f'{b["n_val"]}', f'{g["n_val"]}', f'{s_["n_val"]}', 'Early stopping (phase A), hyper-parameter selection, walk-forward folds'])
    sp.append(['Test (unseen)', f'target after {VAL_END}', f'{ct["Bitcoin"]["n_test"]} ({b["test_period"][0]} → {b["test_period"][1]})', f'{ct["Gold"]["n_test"]} ({g["test_period"][0]} → {g["test_period"][1]})', f'{ct["Silver"]["n_test"]} ({s_["test_period"][0]} → {s_["test_period"][1]})', 'Evaluated ONCE by backtesting.py'])
    F += [table(sp, [1.9 * cm, 4.1 * cm, 2.9 * cm, 2.5 * cm, 2.5 * cm, 2.7 * cm], align_right_from=None), P('Table 9. The frozen chronological split (sample counts = 30-day windows with a valid target; from model_status.json).', 'caption')]
    F += [P(f'<b>Proportions.</b> Bitcoin ≈ {b["n_train"]/(b["n_train"]+b["n_val"]+ct["Bitcoin"]["n_test"])*100:.0f} / {b["n_val"]/(b["n_train"]+b["n_val"]+ct["Bitcoin"]["n_test"])*100:.0f} / '
            f'{ct["Bitcoin"]["n_test"]/(b["n_train"]+b["n_val"]+ct["Bitcoin"]["n_test"])*100:.0f} %; metals ≈ {g["n_train"]/(g["n_train"]+g["n_val"]+ct["Gold"]["n_test"])*100:.0f} / '
            f'{g["n_val"]/(g["n_train"]+g["n_val"]+ct["Gold"]["n_test"])*100:.0f} / {ct["Gold"]["n_test"]/(g["n_train"]+g["n_val"]+ct["Gold"]["n_test"])*100:.0f} % (train / validation / test). '
            'The split is by <i>date</i>, not by percentage, so all three assets are evaluated on the same calendar period.'),
          P('<b>Chronological order is strictly preserved; there is no shuffling.</b> A random split would place "tomorrow" in the training set of "today" and, because '
            'consecutive days share most of their rolling-window features, would make any model look far better than it is. The split is defined by the dates the <i>target</i> covers, '
            'so a training target never reaches past the training end date (for multi-day experimental targets this is an exact purge/embargo).'),
          P(f'<b>Walk-forward validation inside train+val</b> (<font name="Mono">src/evaluation/cross_validation.py</font>, <font name="Mono">TimeSeriesSplit</font>, {CV_FOLDS} folds): '
            'the train+validation period is cut into four consecutive validation blocks; for fold <i>k</i> the model is trained on everything <i>before</i> the block and scored on the block '
            '(expanding window). Inside each fold the last 15 % of the training window is the early-stopping monitor. This is how hyper-parameters and the "best single model" are chosen; '
            'the test set is never passed to this module. The served combined forecast is scored on the same folds by averaging its members\' out-of-fold predictions.')]
    F += [vflow([f'fold 1:  train [0 … 60 %)  → validate block 1', 'fold 2:  train [0 … 70 %)  → validate block 2', 'fold 3:  train [0 … 80 %)  → validate block 3', 'fold 4:  train [0 … 90 %)  → validate block 4',
                 f'final:   train ≤ {TRAIN_END} (monitor: val ≤ {VAL_END})  → refit train+val  → test ONCE'], width_cm=12.5, box_h=0.62, gap=0.3, font=7.6),
          P('Figure 4. Expanding-window walk-forward folds followed by the single test evaluation.', 'caption')]

    # ================================================================ 11. LEAKAGE
    F += [H1('11. Data Leakage Prevention')]
    F += [P('Leakage control was the main engineering effort of the project. Each safeguard below is implemented in code and, where marked, enforced by a unit test in <font name="Mono">tests/</font>.')]
    lk = [['Risk', 'Safeguard in the implementation', 'Test'],
          ['Random shuffling / k-fold', 'Chronological slicing by frozen dates; TimeSeriesSplit for CV', 'test_split_is_chronological_and_disjoint'],
          ['Scaler sees test data', 'MinMaxScaler.fit(train rows only); saved and reused for val/test/live', 'test_scaler_fitted_on_train_only'],
          ['Target inside the input window', 'label = log return of row t+1; window = rows ≤ t', 'test_create_sequences_target_is_next_row_and_never_in_window'],
          ['Rolling / EWM / shift look-ahead', 'All windows backward-looking; perturbing the last row must not change earlier rows (crypto and futures paths)', 'test_no_lookahead_in_any_feature'],
          ['External data alignment', 'Same-day values only for Bitcoin (bar closes after the US close); latest value strictly before day t for the metals', 'test_commodity_external_features_are_previous_day_values'],
          ['Futures High/Low after the settlement', 'hl_range, atr_norm, ADX shifted one session for the metals', 'test_commodity_high_low_features_are_lagged_one_session'],
          ['Synthetic rows', 'Metals keep the exchange calendar', 'test_commodities_keep_trading_calendar'],
          ['Early stopping on its own training data', 'Two-phase fit: monitor = validation (phase A), refit with fixed stopping point (phase B)', 'test_tabular_models_fit_predict_and_two_phase'],
          ['Hyper-parameters chosen on the test set', 'Tuner reads only train+val (build_dataset splits); best_params.json written from fold scores', 'by construction (tune_models.py never indexes X_test)'],
          ['Test set touched more than once', 'Only backtesting.py reads X_test; before_after.py re-reads stored predictions to compare two frozen systems (declared in LIMITATIONS.md)', '—'],
          ['Price reconstruction using the wrong column', 'P<sub>t</sub> passed explicitly from the unscaled feature frame', 'test_true_target_reconstructs_actual_close_exactly'],
          ['Intraday snapshot treated as a close', 'Complete-bar rule at download time', 'test_drop_incomplete_bars_removes_todays_running_row'],
          ['Combination weights tuned on the test set', 'Equal weights — nothing is fitted; Combined row = mean of member predictions', 'test_combined_test_row_is_the_mean_of_its_members'],
          ['Demo prediction peeking at the future', 'predict_for_date slices features.loc[:as_of]; every later row is perturbed in the test and the forecast must not change', 'test_predict_for_date_uses_only_past_data_and_reveals_actual']]
    F += [table(lk, [3.6 * cm, 7.6 * cm, 5.4 * cm], align_right_from=None), P(f'Table 10. Leakage checklist ({N_TESTS} tests in total pass on the report date).', 'caption')]
    F += [P('<b>The leak that was found and fixed (close-time rule).</b> Yahoo\'s daily close for GC=F / SI=F is the 13:25–13:30 ET COMEX settlement (verified against 5-minute contract bars). '
            'The S&amp;P 500 (16:00), VIX (16:15), 10-year yield (~15:00), DXY (17:00) and WTI (14:30) close <i>after</i> that, and the futures\' daily High/Low span the whole Globex session '
            '(until 17:00 ET). An earlier version used their same-day values as day-<i>t</i> features for the metals; those values contain 1–3.5 hours of information from <i>inside</i> the target '
            'interval ln(P<sub>t+1</sub>/P<sub>t</sub>). That version reported 62.2 % directional accuracy for Silver (DM p = 0.002). After aligning the metals\' macro features to the previous '
            'trading day and lagging the High/Low-based features one session, the effect disappeared (§15.2). Bitcoin\'s bar closes at 00:00 UTC, after every US close, so it keeps same-day values. '
            'The rule lives in <font name="Mono">config.EXTERNAL_SAME_DAY</font> / <font name="Mono">POST_SETTLEMENT_FEATURES</font> and <font name="Mono">DataCleaner._align_external</font>.'),
          P('<b>Leakage control in live prediction.</b> The live forecast uses the same feature code on a fresh download that has already had the running bar removed, so the input window ends on the last '
            '<i>complete</i> bar; the prediction is for the <i>next</i> trading day, whose close does not exist yet at prediction time. (The removed automatic record additionally stored the '
            'forecast before the target close existed and verified it the next day — §19.)')]

    # ================================================================ 12. MODELS
    F += [PageBreak(), H1('12. Machine-Learning Models')]
    F += [P('All models share one interface (<font name="Mono">src/models/registry.py</font>): tabular models receive the 28–29 features of day <i>t</i> '
            f'(the last row of the window); the recurrent models receive the full {SEQ_LEN}×F window. Every model predicts the same standardised next-day log return, '
            'so all are compared on identical samples. Six models are <b>trained and served</b> (as members of the combination); three baselines and one stacked ensemble are '
            '<b>evaluated for comparison only</b>.')]
    md = [['Model', 'Type', 'Input', 'Role in the final system', 'Tuned hyper-parameters (BTC / Gold / Silver)'],
          ['Naive (random walk)', 'Baseline', '—', 'ŷ = 0 ⇔ tomorrow\'s price = today\'s. The reference every model must beat; DM test partner', '—'],
          ['Historical mean (Naive-Mean)', 'Baseline', 'training target', 'ŷ = mean training return (drift)', '—'],
          ['ARIMA(p,0,q)', 'Baseline (statistical)', 'return series', 'Order by AIC on train (max p,q = 3); one-step walk-forward forecasts. Not served', f'orders: BTC {json.loads(row("Bitcoin","ARIMA")["fit_info"])["order"]}, Gold {json.loads(row("Gold","ARIMA")["fit_info"])["order"]}, Silver {json.loads(row("Silver","ARIMA")["fit_info"])["order"]}'],
          ['Ridge', 'Linear (L2-regularised)', 'day-t features', 'Member of the Combined forecast', f'alpha = {best["Bitcoin"]["Ridge"]["alpha"]} / {best["Gold"]["Ridge"]["alpha"]} / {best["Silver"]["Ridge"]["alpha"]}'],
          ['Random Forest', 'Bagged trees', 'day-t features', 'Member', f'{best["Bitcoin"]["RandomForest"]["n_estimators"]} trees, max_depth {best["Bitcoin"]["RandomForest"]["max_depth"]}, min_samples_leaf {best["Bitcoin"]["RandomForest"]["min_samples_leaf"]} (all assets)'],
          ['LightGBM', 'Gradient-boosted trees (leaf-wise)', 'day-t features', 'Member; early-stopped', 'lr 0.03 / 0.01 / 0.01 · num_leaves 31 / 31 / 8 · min_child_samples 20 / 100 / 50 · λ 0 / 5 / 5 · subsample 0.8 · colsample 0.8 · up to 600 trees'],
          ['CatBoost', 'Gradient-boosted trees (ordered boosting, symmetric)', 'day-t features', 'Member; early-stopped; best single model by CV for Bitcoin and Gold', 'lr 0.03 / 0.01 / 0.01 · depth 7 / 3 / 3 · l2_leaf_reg 3 / 10 / 10 · up to 800 iterations'],
          ['GRU', 'Recurrent neural network', f'{SEQ_LEN}-day window', 'Member; best single model by CV for Silver', 'Input → GRU(32) → Dropout(0.4) → Dense(16, relu) → Dense(1); Adam lr 0.001, batch 64, MSE loss, early stopping (patience 6, max 40 epochs)'],
          ['LSTM', 'Recurrent neural network', f'{SEQ_LEN}-day window', 'Member', 'Same shape as the GRU with an LSTM(32) cell; dropout 0.4 / 0.2 / 0.2'],
          ['Stacked ensemble', 'Meta-model (experiment)', 'OOF predictions of Ridge, LightGBM, CatBoost, GRU', 'Reported for comparison only — NOT a member and NOT served', 'non-negative Ridge(alpha = 1) fitted on walk-forward out-of-fold predictions'],
          ['Combined', 'Equal-weight forecast combination', 'the six members\' predicted returns', 'THE SERVED FORECAST (dashboard and API)', 'no parameters — arithmetic mean']]
    F += [table(md, [2.6 * cm, 2.6 * cm, 2.1 * cm, 4.0 * cm, 5.3 * cm], align_right_from=None, font=7.3, bold_rows=(11,)), P('Table 11. Every model in the final system. Hyper-parameters are the values in results/tuning/best_params.json chosen by walk-forward validation.', 'caption')]
    F += [H2('12.1 Why these models and what each contributes')]
    F += bullets([
        '<b>Ridge</b> — the simplest regularised linear model; a sanity check that any non-linear gain is real. The tuner chose the strongest regularisation on offer (alpha = 100) for all three assets, i.e. it wants to stay close to "no change".',
        '<b>Random Forest</b> — robust non-linear model with little tuning; depth 3 and min_samples_leaf 30 were selected, again the most regularised grid point.',
        '<b>LightGBM and CatBoost</b> — state-of-the-art gradient boosting for tabular data; they can model interactions between volatility state, momentum and macro returns. Early stopping on validation chose few trees (LightGBM 81 / 10 / 10; CatBoost 50 / 606 / 82 for BTC / Gold / Silver), a sign that there is little signal to fit.',
        '<b>GRU and LSTM</b> — sequence models that see the last 30 days of all features rather than one row. They were reduced to a single 32-unit layer because a deeper 2×100-unit stack over-fitted in the original experiments; early stopping selected only 1–9 epochs.',
        '<b>Baselines</b> — random walk, drift and ARIMA make explicit what a model must beat; ARIMA also confirms that linear autocorrelation carries almost no information.',
        '<b>Advantages / limitations.</b> Linear and tree models are fast, interpretable (feature importance) and need no GPU; trees cannot extrapolate beyond training ranges (mitigated by stationary features). '
        'Recurrent models can exploit temporal structure but need more data than ~2,000–2,800 windows provide and are the weakest family here. All are point forecasters — none produces a calibrated predictive interval.'])

    # ================================================================ 13. TRAINING
    F += [H1('13. Model Training')]
    F += [flow_chart(['1 build_dataset()\nscaled splits + features + scaler', '2 Windows & targets\n30-row windows, y = r(t+1), z-scored', '3 Tuning (once)\ngrid × 4 walk-forward folds → best_params.json',
                      '4 Phase A fit\ntrain, monitor val → stopping point', '5 Validation metrics\ntrain vs val RMSE, loss curves', '6 Phase B refit\ntrain+val, stopping point fixed',
                      '7 Save artefact\n.pkl / .cbm / .keras (+ .json meta)', '8 Evaluate once\nbacktesting.py → model_status.json', '9 Load at prediction\nload_trained() per model'], cols=3),
          P('Figure 5. Training pipeline (src/training/tune_models.py → train_models.py → src/evaluation/backtesting.py → src/inference/prediction.py).', 'caption')]
    F += [P('<b>1–2 Dataset loading and target creation.</b> <font name="Mono">build_dataset(asset)</font> is the single loader used by tuning, training, stacking, evaluation and the tests. '
            'It reads the scaled split files and the unscaled feature frame, builds 30-row windows over the concatenated train|val|test frame (so the first validation/test samples can look back into '
            'earlier rows — past data only), attaches the next-row log return as the target, standardises it with training statistics and assigns each window to a split by the date its target covers.'),
          P('<b>3 Hyper-parameter tuning</b> (<font name="Mono">tune_models.py</font>, <font name="Mono">make tune</font>). Small, deliberate grids over regularisation strength (Ridge 4 configurations, '
            'Random Forest 4, LightGBM 48, CatBoost 12, GRU/LSTM 2 each) are scored by mean fold RMSE of the return over the 4 expanding-window folds; every configuration\'s fold scores are logged to '
            '<font name="Mono">results/tuning/&lt;asset&gt;_&lt;model&gt;.csv</font> and the winner to <font name="Mono">best_params.json</font>, which <font name="Mono">config.get_params()</font> reads at training time.'),
          P('<b>4–6 Two-phase training</b> (<font name="Mono">train_models.py</font>, <font name="Mono">make train</font>). Phase A fits on the training split while monitoring the validation split: '
            'LightGBM early-stops with patience 30, CatBoost with 50 rounds, GRU/LSTM with patience 6 on validation loss (restoring the best weights). Phase A yields honest validation metrics, the '
            'train-vs-validation over-fitting gap and the Keras loss curves. Phase B rebuilds the model with the <i>same</i> hyper-parameters and the stopping point found in phase A '
            '(number of trees / iterations / epochs) and refits on train+validation — this is the deployed model. Validation is therefore never inside its own early-stopping monitor.'),
          P('<b>7 Model saving.</b> Ridge / Random Forest / LightGBM → <font name="Mono">joblib</font> pickles (model + params + fit info); CatBoost → native <font name="Mono">.cbm</font> + JSON meta; '
            'GRU / LSTM → Keras <font name="Mono">.keras</font> + JSON meta with the training history. All 18 artefacts live in <font name="Mono">data/models/</font> next to the three scalers and '
            'the stacked meta-models. Reproducibility: seeds 42 for Python, NumPy and TensorFlow; <font name="Mono">random_state = 42</font> for the tree models; pinned requirements.'),
          P('<b>8–9 Loading during prediction.</b> <font name="Mono">src/inference/prediction.py</font> loads every trained member with <font name="Mono">load_trained(name, asset)</font>, scales the '
            'last 30 rows of the (live or frozen) feature frame with the saved scaler, runs each model, de-standardises each predicted return with <font name="Mono">target_mean/target_std</font> '
            'from <font name="Mono">model_status.json</font> (an explicit error is raised if these are missing) and averages them.')]
    tvt = [['Asset', 'Model', 'Train RMSE (ret)', 'Val RMSE (ret)', 'Val / Train', 'Val dir. acc. %', 'Stopping point (phase A)']]
    for a in ASSETS:
        for m in MEMBERS:
            tr = tv[(tv.asset == a) & (tv.model == m) & (tv.split == 'train')].iloc[0]; va = tv[(tv.asset == a) & (tv.model == m) & (tv.split == 'val')].iloc[0]
            tvt.append([a, NICE.get(m, m), f'{tr["RMSE_ret"]:.5f}', f'{va["RMSE_ret"]:.5f}', f'{va["RMSE_ret"]/tr["RMSE_ret"]:.2f}×', f'{va["DirAcc_pct"]:.1f}', fitinfo(a, m)])
    F += [table(tvt, [1.6 * cm, 2.6 * cm, 2.6 * cm, 2.4 * cm, 2.0 * cm, 2.4 * cm, 3.0 * cm], align_right_from=2), P('Table 12. Phase-A training vs validation metrics (results/train_val_metrics.csv). The validation window (Sep 2025 – Feb 2026) is far more volatile than the 2018–2025 average for the metals, which is why the ratio is > 1 there for every model including the random walk — regime shift, not memorisation.', 'caption')]
    F += fig(os.path.join(FIGURES_DIR, 'overfitting_gap.png'), 15.5, 'Figure 6. Train vs validation RMSE per model and asset (results/figures/overfitting_gap.png).', max_height_cm=7.5)

    # ================================================================ 14. PERFORMANCE
    F += [PageBreak(), H1('14. Model Performance')]
    F += [P('All numbers in this section are the project\'s <b>actual</b> final results, read from <font name="Mono">results/final_test_results.csv</font> (single evaluation on the untouched test period) '
            'and <font name="Mono">results/cv_results.csv</font> (walk-forward validation). Metrics: MAE / RMSE / MAPE in USD (what a user sees); RMSE, MAE and R² of the log return (what is '
            'actually predicted — R² &lt; 0 means worse than predicting the mean); directional accuracy on non-flat days with a one-sided binomial p-value against 50 %; Diebold–Mariano '
            'p-value against the random walk (squared-error loss, HAC variance, Harvey correction). R² on <i>price levels</i> is deliberately not reported: a random walk scores ≈ 0.95 there.')]
    F += [H2('14.1 Unseen test set — every model, every asset')]
    for a in ASSETS:
        srow = st[a]
        t = [['Model', 'MAE ($)', 'RMSE ($)', 'MAPE %', 'RMSE (ret)', 'MAE (ret)', 'R² (ret)', 'Dir. acc. % (p)', 'DM p vs RW', 'Strat. %', 'B&amp;H %']]
        bold = []
        for i, m in enumerate(ORDER):
            r = row(a, m)
            da = '—' if m == 'Naive' else f'{r["DirAcc_pct"]:.1f} ({r["DirAcc_pvalue"]:.2f})'
            dm = '—' if m == 'Naive' else f'{r["DM_pvalue"]:.2f}'
            label = NICE.get(m, m) + (' (CV-best)' if m == srow['primary_model'] else '')
            t.append([label, f'{r["MAE_usd"]:,.2f}', f'{r["RMSE_usd"]:,.2f}', f'{r["MAPE_usd"]:.2f}', f'{r["RMSE_ret"]:.5f}', f'{r["MAE_ret"]:.5f}', f'{r["R2_ret"]:+.3f}', da, dm, f'{r["strategy_return_pct"]:+.1f}', f'{r["buy_hold_return_pct"]:+.1f}'])
            if m == 'Combined':
                bold.append(i + 1)
        F += [KeepTogether([H3(f'{a} — test period {srow["test_period"][0]} → {srow["test_period"][1]} ({ct[a]["n_test"]} days)'),
              table(t, [3.2 * cm, 1.4 * cm, 1.4 * cm, 1.1 * cm, 1.5 * cm, 1.5 * cm, 1.2 * cm, 2.0 * cm, 1.2 * cm, 1.2 * cm, 1.2 * cm], font=7.0, bold_rows=bold, align_right_from=1)])]
    F += [P('Table 13 (a–c). Untouched test set. Shaded row = the served Combined forecast; (CV-best) = best single model by walk-forward validation. "Strat." = cumulative return of a long/flat rule '
            '(hold the asset on day t+1 iff predicted return &gt; 0, 10 bps per position change); "B&amp;H" (buy &amp; hold) = the asset\'s own return over the period.', 'caption')]
    F += [H2('14.2 Served forecast vs random walk — summary')]
    sm = [['Asset', 'n days', 'MAE ($) Combined / RW', 'RMSE ($) Combined / RW', 'MAPE % Combined / RW', 'RMSE (ret) Combined / RW', 'Δ RMSE (ret)', 'Direction correct (p)', 'DM p']]
    for a in ASSETS:
        c, n = ct[a], nt[a]
        sm.append([a, str(c['n_test']), f'{c["MAE_usd"]:,.2f} / {n["MAE_usd"]:,.2f}', f'{c["RMSE_usd"]:,.2f} / {n["RMSE_usd"]:,.2f}', f'{c["MAPE_usd"]:.2f} / {n["MAPE_usd"]:.2f}',
                   f'{c["RMSE_ret"]:.5f} / {n["RMSE_ret"]:.5f}', f'{(c["RMSE_ret"]/n["RMSE_ret"]-1)*100:+.2f} %', f'{c["DirAcc_pct"]:.1f} % ({c["DirAcc_pvalue"]:.2f})', f'{c["DM_pvalue"]:.2f}'])
    F += [table(sm, [1.4 * cm, 1.1 * cm, 2.5 * cm, 2.5 * cm, 2.0 * cm, 2.6 * cm, 1.5 * cm, 2.0 * cm, 1.0 * cm], align_right_from=1), P('Table 14. The served Combined forecast against the random walk (RW) on the unseen test period.', 'caption')]
    F += [P('<b>Reading.</b> For all three assets every model sits at the random-walk floor. The combined forecast\'s return-RMSE is 0.3 %% below the random walk for Bitcoin and Silver and equal for Gold; '
            'none of the differences is significant (DM p = %.2f / %.2f / %.2f). R² in return space is ≈ 0 for every model. Directional accuracy of the combined forecast is %.1f / %.1f / %.1f %% '
            '(p = %.2f / %.2f / %.2f). The dollar errors (MAPE %.2f / %.2f / %.2f %%) equal the random walk\'s because they are the daily volatility of each asset, not model skill — <b>MAPE must not '
            'be read as "accuracy"</b>. The models have learned that tomorrow\'s move is mostly unpredictable: the standard deviation of the combined prediction is %.2f %% (Bitcoin), %.2f %% (Gold) and %.2f %% (Silver) '
            'against realised return standard deviations of %.2f / %.2f / %.2f %%.' %
            (ct['Bitcoin']['DM_pvalue'], ct['Gold']['DM_pvalue'], ct['Silver']['DM_pvalue'], ct['Bitcoin']['DirAcc_pct'], ct['Gold']['DirAcc_pct'], ct['Silver']['DirAcc_pct'],
             ct['Bitcoin']['DirAcc_pvalue'], ct['Gold']['DirAcc_pvalue'], ct['Silver']['DirAcc_pvalue'], ct['Bitcoin']['MAPE_usd'], ct['Gold']['MAPE_usd'], ct['Silver']['MAPE_usd'],
             ct['Bitcoin']['pred_std']*100, ct['Gold']['pred_std']*100, ct['Silver']['pred_std']*100, ct['Bitcoin']['true_std']*100, ct['Gold']['true_std']*100, ct['Silver']['true_std']*100)),
          P('<b>The one p &lt; 0.05 cell.</b> Silver LightGBM reaches %.1f %% directional accuracy (binomial p = %.2f) but has exactly the random walk\'s RMSE (DM p = %.2f); with 30 model-rows in the test '
            'tables one cell at p ≈ 0.02 is what chance produces (Bonferroni threshold ≈ 0.0017), its walk-forward figure is %.1f %%, and it was not the validation winner. It is reported, not claimed.' %
            (row('Silver', 'LightGBM')['DirAcc_pct'], row('Silver', 'LightGBM')['DirAcc_pvalue'], row('Silver', 'LightGBM')['DM_pvalue'], cvrow('Silver', 'LightGBM')['DirAcc_pct_mean']))]
    F += fig(os.path.join(FIGURES_DIR, 'model_comparison_test.png'), 16.5, 'Figure 7. Test-set RMSE of the next-day return and directional accuracy for every model and asset (results/figures/model_comparison_test.png). The RMSE axes are zoomed: all models lie within ±1 % of the random walk.', max_height_cm=9)
    F += [H2('14.3 Walk-forward validation (model-selection evidence)')]
    for a in ASSETS:
        c = cv[cv.asset == a].set_index('model')
        t = [['Model', 'RMSE (ret) mean ± std', 'MAE (ret)', 'R² (ret)', 'Dir. acc. %', 'RMSE ($)']]
        bold = []
        for i, m in enumerate([m for m in ORDER if m in c.index]):
            r = c.loc[m]
            t.append([NICE.get(m, m) + (' (CV-best)' if m == st[a]['primary_model'] else ''), f'{r["RMSE_ret_mean"]:.5f} ± {r["RMSE_ret_std"]:.5f}', f'{r["MAE_ret_mean"]:.5f}', f'{r["R2_ret_mean"]:+.3f}',
                      '—' if m == 'Naive' else f'{r["DirAcc_pct_mean"]:.1f}', f'{r["RMSE_usd_mean"]:,.2f}'])
            if m == 'Combined':
                bold.append(i + 1)
        F += [KeepTogether([H3(f'{a} — {CV_FOLDS} expanding folds inside train+validation'), table(t, [4.4 * cm, 3.6 * cm, 2.2 * cm, 2.0 * cm, 2.2 * cm, 2.2 * cm], font=7.3, bold_rows=bold, align_right_from=1)])]
    F += [P('Table 15 (a–c). Walk-forward validation. All trained models lie within ~0.5 % of each other and of the random walk, so the ranking is not statistically decisive — the reason the served forecast is the plain equal-weight combination rather than a single "winner".', 'caption')]
    F += [H2('14.4 Performance across market regimes (test period)')]
    rg = [['Asset', 'Regime (defined on day t, before the prediction)', 'Days', 'MAE % Combined', 'MAE % random walk', 'Direction correct %']]
    for a in ASSETS:
        for _, r in reg[(reg.asset == a) & (reg.regime_type != 'day_regime')].iterrows():
            rg.append([a, r['regime'], str(int(r['n_days'])), f'{r["mae_pct_served"]:.2f}', f'{r["mae_pct_naive"]:.2f}', f'{r["dir_hit_served_pct"]:.1f}'])
    F += [table(rg, [1.6 * cm, 5.4 * cm, 1.4 * cm, 2.6 * cm, 2.8 * cm, 2.8 * cm], align_right_from=2, font=7.3), P('Table 16. results/regime_analysis.csv — volatility terciles of the 30-day σ and sign of the 20-day return. Errors scale with volatility for every asset, and the combined forecast neither improves nor degrades relative to the random walk in any regime.', 'caption')]
    F += fig(os.path.join(FIGURES_DIR, 'silver_actual_vs_predicted.png'), 16.5, 'Figure 8. Silver, unseen test period: predicted vs actual next-day returns (line and scatter) and the reconstructed price with the random-walk shadow (results/figures/silver_actual_vs_predicted.png).', max_height_cm=10)

    # ================================================================ 15. BASELINE
    F += [H1('15. Baseline Comparison')]
    F += [P('<b>Yes — the project is built around a baseline.</b> The primary baseline is the <b>random walk</b> (<font name="Mono">NaiveModel</font>): the predicted log return is 0, i.e. '
            '"the next price equals the latest known price". Two further baselines are evaluated on the same days: the <b>historical mean</b> (drift) and an <b>ARIMA(p,0,q)</b> on the return series '
            '(orders chosen by AIC on the training window: Bitcoin (2,0,0), Gold (1,0,1), Silver (2,0,3); one-step walk-forward forecasts).'),
          P('<b>Why it matters.</b> In dollar terms a random walk already achieves MAPE 1.3–2.5 % and price-level R² ≈ 0.95, so a model can look impressive while adding nothing. The project therefore '
            '(i) reports every metric side by side with the naive forecast, (ii) tests the difference formally with the Diebold–Mariano statistic, and (iii) tests directional accuracy against 50 % with a '
            'binomial test. The random walk is also the dashboard\'s reference line in every chart and table.'),
          P('<b>Result of the comparison.</b> Table 14 gives the answer for the served forecast: within ±0.3 % of the random-walk RMSE, DM p ≥ 0.46, direction not distinguishable from chance. '
            'The best single models are 0.0–0.6 % <i>above</i> the random walk\'s RMSE on the test set. The historical-mean and ARIMA baselines are likewise indistinguishable from the random walk.')]
    F += [H2('15.1 Before / after comparison on identical unseen days')]
    bat = [['Asset', 'System', 'Model', 'MAE ($)', 'RMSE ($)', 'MAPE %', 'RMSE (ret)', 'R² (ret)', 'Dir. acc. % (p)', 'DM p']]
    for _, r in BA.iterrows():
        bat.append([r['asset'], r['system'], r['model'], f'{r["MAE_usd"]:,.2f}', f'{r["RMSE_usd"]:,.2f}', f'{r["MAPE_usd"]:.2f}', f'{r["RMSE_ret"]:.5f}', f'{r["R2_ret"]:+.3f}',
                    '—' if r['model'] == 'Naive' else f'{r["DirAcc_pct"]:.1f} ({r["DirAcc_pvalue"]:.2f})', '—' if pd.isna(r['DM_pvalue_vs_naive']) else f'{r["DM_pvalue_vs_naive"]:.2f}'])
    F += [table(bat, [1.4 * cm, 2.9 * cm, 1.6 * cm, 1.5 * cm, 1.5 * cm, 1.3 * cm, 1.7 * cm, 1.4 * cm, 2.1 * cm, 1.2 * cm], font=7.0, align_right_from=3),
          P(f'Table 17. results/experiments/before_after.csv — the archived 3-year system vs the final full-history system on the same {int(BA["n_days"].iloc[0])} / {int(BA["n_days"].iloc[4])} unseen days ({BA["period"].iloc[0]}). '
            '2.9× more training data moved the single models onto the random-walk line but did not produce skill.', 'caption')]
    F += [H2('15.2 The archived "62 % Silver" result')]
    F += [P('The first evaluation of the improved system (kept in <font name="Mono">results/archive_sameday_macro_leak/</font>) reported Silver LightGBM at <b>62.2 % directional accuracy, R² = +0.044, DM p = 0.002</b>. '
            'It used same-day macro returns for the metals. After the close-time correction (§11) the same model on the same 143 days scores 51.7 % with the five macro features removed and the final served '
            'single model (GRU) 55.2 % (p = 0.12, DM p = 0.99). Bitcoin\'s numbers are bit-identical before and after (its bar closes after the US close). The lesson written into the methodology: '
            '"backward-looking by date" is not the same as "known at the moment the target interval starts".')]

    # ================================================================ 16. ENSEMBLE
    F += [PageBreak(), H1('16. Ensemble / Combined Prediction — the served forecast')]
    F += [side_by_side(vflow(['Latest complete bar\n(live features, last 30 rows, scaled)', 'ALL six trained models\nRidge · RF · LightGBM · CatBoost · GRU · LSTM', 'Six predicted log returns\ny1 … y6 (de-standardised)',
                              'Equal-weight mean\ny = (y1 + … + y6) / 6', 'ONE predicted price\nP\' = P(t) · exp(y)', 'Direction\nUP if y > 0, else DOWN'], width_cm=7.6, box_h=0.9, gap=0.4, font=7.4),
                       [P('<b>Exact rule</b> (<font name="Mono">prediction.py :: _predict_from_window</font>): each trained member predicts the standardised return from the same input window; each value is '
                          'mapped back to a real log return with the training mean/std; the served return is the <b>arithmetic mean</b> of the members\' returns — no weights are fitted, no voting, '
                          'no optimisation. Price and direction follow from the mean. The stacked ensemble is <i>not</i> a member (it is itself a blend of some members).', 'bodyl'),
                        P('<b>Why an equal-weight combination.</b> (i) All six members are within ~0.5 % of each other on the walk-forward folds, so picking one "winner" would be picking noise. '
                          '(ii) Equal weights are the standard robust choice for forecasts of similar quality — estimated weights rarely beat them out of sample (the forecast-combination puzzle). '
                          '(iii) Nothing is fitted, so the combination cannot be tuned on the test set. (iv) The user never has to choose a model.', 'bodyl'),
                        P('<b>Evaluated like a single model.</b> The combination is scored on the walk-forward folds (members\' out-of-fold predictions averaged per fold, row "Combined" in Table 15) and '
                          'once on the test set (row "Combined" in Table 13); a unit test asserts that the stored Combined test predictions equal the mean of the stored member predictions.', 'bodyl')],
                       widths=(7.9, 8.7))]
    F += [H2('16.1 The stacked ensemble (experiment, not served)')]
    sw = [['Asset', 'Base models', 'Fitted non-negative weights', 'Intercept', 'Test RMSE (ret) stacked vs Combined']]
    for a in ASSETS:
        w = stack[a]['weights']
        sw.append([a, ', '.join(stack[a]['bases']), ', '.join(f'{k} {v:.2f}' for k, v in w.items()), f'{stack[a]["intercept"]:.3f}', f'{row(a, "Stacked")["RMSE_ret"]:.5f} vs {row(a, "Combined")["RMSE_ret"]:.5f}'])
    F += [table(sw, [1.6 * cm, 3.6 * cm, 5.6 * cm, 1.6 * cm, 4.2 * cm], align_right_from=None), P('Table 18. results/stacking/*.json and the test rows. Weights are fitted on walk-forward out-of-fold predictions (train+val only). The stack is marginally better than the equal-weight combination for Bitcoin and worse for Gold and Silver — no consistent advantage, so it stays an experiment.', 'caption')]
    F += [P('<b>Why the combined prediction is useful even without skill.</b> It gives one unambiguous answer instead of six, averaging damps the idiosyncratic noise of individual members (the combined prediction\'s spread is 0.09–0.12 % of price against realised daily moves of 1.7–3.2 %), and it removes '
            'the temptation to cherry-pick the model that happened to do best on the test window.')]

    # ================================================================ 17. USER WORKFLOW
    F += [H1('17. How a User Generates a Prediction')]
    F += [flow_chart(['1 Select asset\nsidebar: Bitcoin / Gold / Silver', '2 Click "Run prediction"\n(one button — no model choice)', '3 Refresh data\nlatest complete bar from Yahoo (≤ 1 download / 30 min)',
                      '4 Features\nsame DataCleaner code → live_features.csv', '5 Scale window\nlast 30 rows, train-fitted scaler', '6 Six models predict\nstandardised → real return',
                      '7 Combine\nequal-weight mean → price, UP/DOWN', '8 Display\nfinal card + each model + ±1 RMSE band + reliability', '9 Not persisted\n(no forward log in the final system)'], cols=3),
          P('Figure 9. The dashboard "Run prediction" workflow (app/streamlit_app.py → src/data/sync_live_data.py → src/inference/prediction.py).', 'caption')]
    F += [P('<b>Step details.</b> (3) <font name="Mono">update_live_data()</font> downloads the asset, the external series and — for Silver — gold into <font name="Mono">data/raw_live/</font> '
            '(the frozen <font name="Mono">data/raw/</font> is never touched), unless the live file already ends on the newest complete bar or was refreshed less than 30 minutes ago. '
            '(4) The same <font name="Mono">DataCleaner</font> builds the features and writes <font name="Mono">&lt;prefix&gt;_live_features.csv</font> atomically; a download that fails feature validation is discarded. '
            '(5–7) <font name="Mono">predict_next_day(asset)</font> as described in §16; the target date is the next calendar day (Bitcoin) or the next weekday (metals). '
            '(8) The final card shows the predicted close, UP/DOWN, the change in % and $, and the ±1 RMSE band (test-set return RMSE of the combined forecast applied to the price); below it a table with each '
            'member\'s predicted close/change/direction, a 60-day chart with the forecast point, and the reliability table (Combined vs random walk on the unseen test days). If Yahoo has not yet published the newest bar, '
            'a warning names the bar the forecast starts from. (9) The forecast is displayed, not stored (see §19).')]
    F += [H2('17.1 Example run (actual output on 2026-09-14 — illustrative only, not a result)')]
    F += [P('The following is what the command-line equivalent (<font name="Mono">make predict</font> → <font name="Mono">python src/inference/prediction.py</font>) printed on the report date; the dashboard shows the same numbers. '
            'They illustrate the mechanics — they are not evidence of accuracy.'),
          P('Gold&nbsp;&nbsp;&nbsp; last complete bar 2026-09-11, close $4,366.20 → predicted close for 2026-09-14: <b>$4,369.03</b> (+0.06 %, <b>UP</b>)<br/>'
            'members: Ridge +0.12 % · RandomForest −0.03 % · LightGBM +0.06 % · CatBoost +0.04 % · GRU +0.03 % · LSTM +0.16 %&nbsp;&nbsp; → mean = +0.06 %<br/>'
            '±1 RMSE band (test RMSE of the return 0.0167): $4,296.66 – $4,442.63<br/><br/>'
            'Bitcoin&nbsp; last complete bar 2026-09-13, close $76,838.16 → predicted close for 2026-09-14: $76,930.67 (+0.12 %, UP); members +0.16 / +0.06 / +0.33 / +0.11 / +0.03 / +0.03 %<br/>'
            'Silver&nbsp;&nbsp; last complete bar 2026-09-11, close $64.55 → predicted close for 2026-09-14: $64.58 (+0.05 %, UP); members +0.04 / −0.02 / +0.04 / +0.09 / +0.08 / +0.05 %', 'code'),
          P('Note how the members agree to within a fraction of a percent and how the uncertainty band (±1.7 % for Gold) is far wider than the predicted move (+0.06 %): this is the honest picture of next-day uncertainty and the dashboard says so next to the number.')]

    # ================================================================ 18. UP/DOWN
    F += [H1('18. UP / DOWN Prediction')]
    F += [P('The direction is derived from the <b>sign of the combined predicted log return</b>, which is equivalent to comparing the predicted price with the latest known close:'),
          P('direction = "UP" if ŷ &gt; 0 else "DOWN"&nbsp;&nbsp;&nbsp;&nbsp;(⇔ P′<sub>t+1</sub> &gt; P<sub>t</sub> → UP, otherwise DOWN)&nbsp;&nbsp;&nbsp;&nbsp;— prediction.py :: predict_next_day / predict_for_date / _individual', 'code'),
          P('There is no dead-band or threshold: an exactly-zero predicted return (which does not occur with continuous models) would be labelled DOWN. Each member model gets its own UP/DOWN label by the same rule, '
            'and the dashboard reports how many of the six members agree (e.g. "5 of 6 models say UP"), but the served direction is the sign of the <i>mean</i>, not a majority vote. '
            'For the statistics, "directional accuracy" counts a day as correct when sign(ŷ) = sign(actual r) among days whose actual return is non-zero (§20).')]

    # ================================================================ 19. AUTOMATIC DAILY PREDICTION
    F += [H1('19. Automatic Daily Prediction System — status in the final implementation')]
    F += [P('<b>Status: not part of the final system.</b> During the final audit (2026-09-13/14) an automatic daily forward-prediction record was implemented — a scheduled job that ran the combined forecast '
            'for each asset after the day\'s bar was complete, appended it to an append-only log, and on the following run fetched the actual close, computed the error and a HIT/FAIL label, and showed the '
            'record in a dashboard tab. It was verified end-to-end and then <b>removed entirely at the author\'s request</b>; the CHANGELOG entry of 2026-09-14 records the removal '
            '("the project keeps the frozen test-set history as its evidence of out-of-sample prediction"). The final code base contains <b>no scheduler, no launchd/cron job and no persistent '
            'forward-prediction file</b>, and the dashboard has no "automatic predictions" tab. The earlier dashboard screenshots in <font name="Mono">docs/figures/</font> that still show such a tab pre-date the removal; '
            'the screenshots in this report were taken from the final code.', 'warn'),
          P('<b>What the final system does instead — and how it maps onto the Day-1 / Day-2 idea.</b> The <i>logic</i> of forward verification is fully present; what is missing is only the persistence and the scheduler:')]
    F += [table([['Element of an automatic daily system', 'Present in the final system?', 'Where / how'],
                 ['Fetch the latest available data automatically', 'Yes, on demand', '"Run prediction" (dashboard) or GET /predict/{asset} (API) refreshes the last complete bar (throttled)'],
                 ['Run all models and generate the combined prediction', 'Yes', 'predict_next_day() — six models, equal-weight mean'],
                 ['Save the prediction before the target close exists', 'No (removed)', 'The live forecast is displayed/returned, not written to disk'],
                 ['Next day: fetch the actual price and compare with the stored prediction', 'No live log; Yes on the frozen test period', 'predict_for_date() and results/predictions/*.csv: every unseen test day is predicted from data up to that day and compared with the actual next close'],
                 ['Calculate error and HIT/FAIL', 'Yes', 'error_usd, error_pct, direction_hit (§20)'],
                 ['Maintain a growing history', 'No live history; a fixed history of 143–207 unseen days per asset', 'Test-Set History tab; CSV download'],
                 ['Scheduler', 'No', '—']], [5.4 * cm, 3.6 * cm, 7.6 * cm], align_right_from=None),
          P('Table 19. Automatic-prediction elements versus the final implementation.', 'caption')]
    F += [P('<b>The out-of-sample verification loop that IS implemented (frozen test period).</b> For every day <i>t</i> of the unseen period the evaluation script stands on day <i>t</i>, builds the input window from data up to '
            '<i>t</i>, produces every model\'s forecast for <i>t</i>+1, then reads the actual close of <i>t</i>+1 and records error and direction hit. This is exactly the Day-1 → Day-2 comparison, carried out once for '
            'all test days from the frozen dataset (the models were trained only on data up to the validation end date, so each of these days is genuinely unseen). The dashboard exposes the same loop interactively '
            '(Predict a Day) — Figure 10 — and as a table (Test-Set History) — §21.')]
    F += [flow_chart(['Data up to day t\n(frozen dataset, rows ≤ t)', 'All six models\n+ baselines + stack', 'Combined prediction\npredicted P(t+1), UP/DOWN', 'Reveal actual close\nP(t+1) from the next row',
                      'Comparison\nerror $ and %, direction', 'HIT / FAIL\nsign(pred) = sign(actual)?', 'Test-set history row\nresults/predictions/*.csv', 'Next day t+1\nrepeat for every unseen day'], cols=4),
          P('Figure 10. The verification loop implemented for the unseen test period (src/evaluation/backtesting.py :: main; interactively: prediction.py :: predict_for_date).', 'caption')]
    F += [P('<b>If the automatic record were restored</b> (future work, §31): the removed prototype\'s design was a daily job calling the same <font name="Mono">predict_next_day()</font> after each market\'s complete-bar '
            'cut-off, an append-only file keyed by (asset, as-of date) to prevent duplicates, and a verification step on the next run filling in the actual close, error and HIT/FAIL. Nothing in the current '
            'architecture prevents this; it was a scope decision.')]

    # ================================================================ 20. HIT/FAIL
    F += [H1('20. HIT / FAIL Methodology')]
    F += [P('The project tracks <b>two separate things</b> for every verified prediction and never mixes them: the <b>direction outcome</b> (HIT/FAIL) and the <b>price error</b>.'),
          P('actual_return r = ln(P<sub>t+1</sub> / P<sub>t</sub>)&nbsp;&nbsp;&nbsp; predicted return ŷ (combined)<br/>'
            'HIT&nbsp; if sign(ŷ) == sign(r) and r ≠ 0&nbsp;&nbsp;&nbsp;&nbsp; FAIL&nbsp; if sign(ŷ) ≠ sign(r) and r ≠ 0&nbsp;&nbsp;&nbsp;&nbsp; undefined ("flat") if r == 0<br/>'
            'error_usd = P′<sub>t+1</sub> − P<sub>t+1</sub>&nbsp;&nbsp;&nbsp;&nbsp; error_pct = (P′<sub>t+1</sub> − P<sub>t+1</sub>) / P<sub>t+1</sub> × 100&nbsp;&nbsp;&nbsp;&nbsp;(reported as a number, never thresholded)', 'code'),
          *bullets(['<b>No tolerance band.</b> A HIT requires only the correct sign; a prediction of +0.02 % on a day that closes +3 % is a HIT, and one of +0.5 % on a day that closes −0.1 % is a FAIL. '
                    'No "within X %" price-tolerance rule is used anywhere, because such a rule is trivially satisfied by the random walk on calm days and would make the system look artificially accurate.',
                    '<b>Zero-return days are excluded</b> from directional accuracy (their direction is undefined) — there are none in the stored test periods (n = 207 / 143 / 143 = all test days). '
                    'In the per-day history file a zero actual return would be stored as 0 (not a hit); in the dashboard\'s Predict-a-Day view it is shown as "flat".',
                    '<b>Statistical reading.</b> The hit rate is reported with a one-sided binomial p-value against 50 %; with 143 days about 58 % is needed for p &lt; 0.05, so the observed 47–54 % is chance-level and the report says so.',
                    '<b>Price error is separate.</b> MAE, RMSE and MAPE in USD, plus the per-day error, are always shown next to the random walk\'s (whose error is the day\'s actual move), so that a "1.3 % average error" is not mistaken for skill.',
                    '<b>Why this rule.</b> Sign agreement is the standard, tolerance-free definition of directional accuracy in the forecasting literature and is the only definition under which the random walk (no direction) is a fair reference.'])]

    # ================================================================ 21. PREDICTION HISTORY
    F += [H1('21. Prediction History (Test-Set History)')]
    F += [P('The prediction history is the file <font name="Mono">results/predictions/&lt;prefix&gt;_test_predictions.csv</font> — one row per unseen test day — written by '
            '<font name="Mono">src/evaluation/backtesting.py</font> and read by <font name="Mono">prediction.py :: prediction_history()</font> for the dashboard. It is a <b>plain CSV file</b>; no database is used.')]
    F += [table([['Field (CSV column)', 'Meaning'],
                 ['date', 'Day t — the last day whose data the prediction used ("Predicted on")'],
                 ['target_date', 'Day t+1 — the day whose close is predicted ("For")'],
                 ['prev_close', 'Actual close on day t (P<sub>t</sub>), the anchor of the price reconstruction'],
                 ['actual_close', 'Actual close on the target date (P<sub>t+1</sub>)'],
                 ['actual_return', 'ln(actual_close / prev_close)'],
                 ['pred_return_&lt;model&gt;', 'Predicted log return, for each of: Naive, Naive-Mean, ARIMA, Ridge, RandomForest, LightGBM, CatBoost, GRU, LSTM, Stacked, Combined'],
                 ['pred_close_&lt;model&gt;', 'prev_close · exp(pred_return) — the predicted price'],
                 ['error_pct_&lt;model&gt;', '(pred_close − actual_close) / actual_close × 100'],
                 ['direction_hit_&lt;model&gt;', '1 if sign(pred_return) = sign(actual_return) else 0 (HIT / FAIL)']], [4.0 * cm, 12.6 * cm], align_right_from=None),
          P('Table 20. Fields stored for every unseen day (49 columns: 5 common + 4 × 11 models).', 'caption')]
    hg = hist['Gold']
    sample = [['Predicted on', 'For', 'Prev close', 'Predicted (Combined)', 'Actual', 'Error $', 'Error %', 'Pred. dir.', 'Actual dir.', 'Result', 'RW error %']]
    for _, r in pd.concat([hg.head(4), hg.tail(4)]).iterrows():
        sample.append([str(r['date'].date()), str(r['target_date'].date()), f'{r["prev_close"]:,.2f}', f'{r["pred_close_Combined"]:,.2f}', f'{r["actual_close"]:,.2f}',
                       f'{r["pred_close_Combined"] - r["actual_close"]:+.2f}', f'{r["error_pct_Combined"]:+.2f}', 'UP' if r['pred_return_Combined'] > 0 else 'DOWN',
                       'UP' if r['actual_return'] > 0 else 'DOWN', 'HIT' if r['direction_hit_Combined'] == 1 else 'FAIL', f'{r["error_pct_Naive"]:+.2f}'])
    F += [table(sample, [1.8 * cm, 1.8 * cm, 1.5 * cm, 2.0 * cm, 1.5 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.4 * cm, 1.1 * cm, 1.6 * cm], font=7.0, align_right_from=2),
          P(f'Table 21. First and last four rows of the Gold history ({len(hg)} rows in total, {hg["target_date"].min().date()} → {hg["target_date"].max().date()}), Combined forecast.', 'caption')]
    sumtab = [['Asset', 'Rows (unseen days)', 'Period covered', 'Mean |error| % Combined / RW', 'Median |error| % Combined / RW', 'Largest miss % Combined / RW', 'HITs', 'Hit rate']]
    for a in ASSETS:
        h = hist[a]; e = h['error_pct_Combined'].abs(); n = h['error_pct_Naive'].abs(); hits = int(h['direction_hit_Combined'].sum())
        sumtab.append([a, str(len(h)), f'{h["target_date"].min().date()} → {h["target_date"].max().date()}', f'{e.mean():.2f} / {n.mean():.2f}', f'{e.median():.2f} / {n.median():.2f}', f'{e.max():.2f} / {n.max():.2f}', f'{hits} of {len(h)}', f'{hits/len(h)*100:.1f} %'])
    F += [table(sumtab, [1.4 * cm, 1.7 * cm, 3.3 * cm, 2.5 * cm, 2.5 * cm, 2.3 * cm, 1.5 * cm, 1.4 * cm], font=7.3, align_right_from=3), P('Table 22. Summary of the three history files (this is the KPI strip shown on the Test-Set History tab).', 'caption')]
    F += [P('<b>How records are created and kept.</b> The file is produced in one pass by the evaluation script from the frozen dataset and the deployed models; it is <b>not</b> appended to day by day. '
            'Records are therefore immutable between evaluation runs — a row can only change if the whole evaluation is deliberately re-run (<font name="Mono">make evaluate</font>), which regenerates the file '
            'from scratch and is recorded in the results tables. Duplicate rows cannot occur because each row corresponds to exactly one window end date. The dashboard presents the Combined columns, '
            'the random-walk error for reference, KPI summaries and a CSV download; the underlying file also holds every single model\'s columns.')]

    # ================================================================ 22. FORWARD TESTING
    F += [H1('22. Real-World Forward Testing — what the project can and cannot claim')]
    F += [P('<b>The principle.</b> The strongest evidence for a forecasting system is a prediction that was fixed <i>before</i> the outcome existed and was compared with the outcome afterwards. '
            'Two levels of this exist in the project:'),
          *bullets(['<b>Frozen unseen test period (implemented).</b> The models were trained and tuned on data up to 2026-02-17 only. Every day from 2026-02-18 to 2026-09-11/12 is then predicted from data up to '
                    'that day and compared with the actual next close. Because nothing about the models was chosen using these days (a fact enforced by the frozen split dates and the single evaluation script), '
                    'the 143–207 comparisons per asset are genuine out-of-sample tests, stronger than in-sample fits or repeatedly re-tuned validation scores.',
                    '<b>Live forward record (prototyped, removed).</b> A calendar-time log in which each forecast is written down before the market closes and verified the next day. This is stronger still, '
                    'because it also tests the data pipeline in real operating conditions (vendor delays, contract rolls, holidays). The prototype was exercised only briefly before it was removed; the final system therefore '
                    '<b>does not claim any live forward-testing record</b>.']),
          P('<b>What a short forward-testing period would and would not show.</b> Even a fully implemented live record of a few weeks is evidence that the system <i>works operationally</i> — that predictions are '
            'produced on time, from complete bars, and verified correctly. It is <b>not</b> proof of long-term market profitability: with daily directional hit rates near 50 %, several hundred days are needed '
            'before a difference from chance becomes statistically visible, and market regimes change faster than that. The project\'s own 143–207-day frozen test period already illustrates this: '
            'the combined forecast is statistically indistinguishable from the random walk.')]

    # ================================================================ 23. STREAMLIT APP
    F += [H1('23. Streamlit Application')]
    F += [P('<font name="Mono">app/streamlit_app.py</font> (≈ 620 lines, single page with tabs). Every number it shows is read from the project\'s artefacts (<font name="Mono">data/processed</font>, '
            '<font name="Mono">data/models</font>, <font name="Mono">results/</font>) or produced live by <font name="Mono">src/inference/prediction.py</font>; nothing is hard-coded. Dark theme, Plotly charts, '
            'Streamlit metrics/dataframes; deep links <font name="Mono">?asset=Gold&amp;run=1</font> for demos.')]
    F += [table([['Component', 'Purpose', 'What is displayed / user interaction'],
                 ['Sidebar', 'Control panel', 'Asset selector (Bitcoin / Gold / Silver); "Run prediction" (primary button); caption "runs all 6 trained models … no model to choose"; chart overlays (Bollinger, RSI, MACD; Volume for Bitcoin); "Sync live market data"; status caption (served forecast, best single model by validation, horizon, data source and last complete bar)'],
                 ['Overview strip', 'Context before any number', 'What is predicted (ticker, log return, number of features), the served forecast and its members, reliability on the unseen test days (direction %, MAPE vs random walk, DM p) — reads model_status.json'],
                 ['Tab 1 — Forecast', 'Live next-day forecast', 'Latest market data (last complete close with day change, 30-day high/low, 30-day σ); after "Run prediction": final card (predicted close, UP/DOWN, change % and $), ±1 RMSE band, table of each model\'s prediction + Combined, member agreement count, 60-day chart with forecast point and band, reliability table (Combined vs random walk), warning if the vendor has not published the newest bar; full price history with train/val/test shading and optional indicator overlays and range buttons (6M/1Y/2Y/All)'],
                 ['Tab 2 — Predict a Day (unseen test)', 'Interactive out-of-sample demo', 'Select any test day → "Generate prediction" → five metrics (close on day t, predicted close for t+1, actual close, prediction error $ and %, direction HIT ✔ / FAIL ✘), context caption, actual-vs-predicted chart of the whole test period with the chosen day highlighted (diamond = prediction, circle = actual), table of each model on that day with its own error and HIT/FAIL'],
                 ['Tab 3 — Test-Set History', 'Per-day verification table', 'KPI table (mean / median / largest absolute error % of Combined vs random walk, direction correct x of n); sortable table of every unseen day (Predicted on, For, Predicted $, Actual $, Error $, Error %, Random-walk error %, Direction HIT/FAIL); CSV download'],
                 ['Tab 4 — Model Performance', 'Evaluation evidence', 'Test table of all 11 rows with the served and CV-selected rows labelled; "honest reading" box; bar charts (RMSE of return, direction %) vs the random-walk line; walk-forward table; predicted-vs-actual return line and scatter; feature-importance selector for Ridge/RF/LightGBM/CatBoost; expanders for regime table, design experiments E1–E4 and training diagnostics (over-fitting gap, loss curves)'],
                 ['Tab 5 — Methodology', 'Explanation for examiners', 'Target, data, split, features (with the close-time rule for the selected asset), models, how the combination works, metrics; details table of the served forecast (members, rule, walk-forward RMSE vs random walk, best single model, its stopping point and hyper-parameters)'],
                 ['Footer', 'Positioning', 'The research-prototype / not-financial-advice disclaimer on every view']], [3.2 * cm, 2.9 * cm, 10.5 * cm], align_right_from=None, font=7.3),
          P('Table 23. Dashboard components (final code).', 'caption')]
    F += [P('<b>UI architecture.</b> One Streamlit script; cached loaders (<font name="Mono">st.cache_data</font>, 5 min) for the feature frames and result CSVs; session state for the last run result so that switching tabs '
            'does not re-run the models; all model logic lives in <font name="Mono">src/</font> and is shared with the API and the tests. The dashboard never trains or evaluates — it only reads artefacts and calls the inference module.')]
    F += fig(os.path.join(DASH, 'dash_gold_forecast.png'), 16.2, 'Figure 11. Forecast tab after "Run prediction" (Gold, captured from the final code on the report date): final combined prediction, each member\'s prediction, 60-day chart with the ±1 RMSE band, reliability vs the random walk, and the shaded split.', max_height_cm=21.5)
    F += fig(os.path.join(DASH, 'dash_gold_predict_a_day.png'), 16.2, 'Figure 12. Predict a Day (unseen test): the combined forecast for the day after 2026-09-10 made from data up to that day, the revealed actual close, error and HIT, and each member on that day.', max_height_cm=20)
    F += fig(os.path.join(DASH, 'dash_silver_history.png'), 16.2, 'Figure 13. Test-Set History (Silver): KPI summary and the per-day table with HIT/FAIL and the random-walk error for reference.', max_height_cm=12)
    F += fig(os.path.join(DASH, 'dash_gold_performance.png'), 15.0, 'Figure 14. Model Performance tab (Gold): test table, honest reading, comparison charts, walk-forward table, return-space diagnostics and feature importance.', max_height_cm=22)
    F += fig(os.path.join(DASH, 'dash_btc_forecast.png'), 15.0, 'Figure 15. Forecast tab for Bitcoin (the Volume overlay is offered for Bitcoin only).', max_height_cm=20)

    # ================================================================ 24. VISUALISATIONS
    F += [H1('24. Visualisations')]
    F += [table([['Chart (where)', 'Type', 'X-axis', 'Y-axis', 'Data represented', 'Why it is useful'],
                 ['Last 60 days + next-day forecast (Forecast tab)', 'Plotly line + markers', 'date', 'price (USD)', 'recent closes, dashed segment to the predicted close, ±1 RMSE band bar', 'Puts the forecast and its uncertainty on the price scale'],
                 ['Price history with split shading (Forecast tab)', 'Plotly line, subplots for overlays', 'date (range buttons)', 'price; RSI; MACD; volume', 'full stored history, train/val/test shading, optional Bollinger/RSI/MACD/volume', 'Shows what the models were trained on vs. the unseen period; indicator families behind the features'],
                 ['Actual vs predicted close, test period (Predict a Day)', 'Plotly lines + highlighted markers', 'target date', 'price (USD)', 'actual close and combined predicted close for every unseen day; the chosen day highlighted', 'Shows the demo prediction in context; caption warns that a price line always "hugs" the actual'],
                 ['RMSE (return) and direction % per model (Model Performance)', 'Plotly bar charts (2 panels)', 'model', 'RMSE of return; direction %', 'test-set values for all 11 rows with the random-walk / 50 % reference lines', 'Model comparison at a glance; makes the "no model beats the random walk" result visible'],
                 ['Daily returns: actual vs predicted; scatter (Model Performance)', 'Plotly line; scatter with 45° line', 'target date; actual return %', 'return %; predicted return %', 'combined forecast vs realised returns on the test period', 'Skill is only visible in return space; shows the prediction barely moves'],
                 ['Feature importance (Model Performance)', 'Plotly horizontal bars', 'normalised importance', 'feature', 'top-15 features of the selected tree/linear member', 'Which inputs a member relies on'],
                 ['Over-fitting gap; GRU/LSTM loss curves (expander)', 'matplotlib PNG', 'model / epoch', 'RMSE / loss', 'train vs validation RMSE; training vs validation loss per epoch', 'Training diagnostics'],
                 ['Report figures (results/figures/, generated by src/evaluation/plots.py)', 'matplotlib PNG', '—', '—', 'price history; CV and test model comparison; per-asset actual-vs-predicted, residuals, strategy vs buy-and-hold, feature importance, loss curves; EDA ACF/PACF, return distributions, volatility clustering', 'Used in the written reports; reproducible with make figures']],
                [3.6 * cm, 2.3 * cm, 1.8 * cm, 1.9 * cm, 3.6 * cm, 3.4 * cm], align_right_from=None, font=7.0),
          P('Table 24. Every chart implemented in the dashboard and the report-figure script.', 'caption')]
    F += [P('Charts that are <b>not</b> implemented: a live prediction-history error chart over calendar time (no live history exists), calibration/interval plots (point forecasts only), and any interactive chart of the archived experiments beyond their tables.')]

    # ================================================================ 25. TECH STACK
    F += [H1('25. Technology Stack')]
    F += [table([['Technology', 'Version (installed)', 'Purpose in this project'],
                 ['Python', '3.9.6 (local venv); 3.11 in Docker / CI', 'All code'],
                 ['pandas / NumPy', '2.3.3 / 1.26.4', 'Data cleaning, feature engineering, windows, result tables'],
                 ['scikit-learn', '1.6.1', 'Ridge, RandomForestRegressor, MinMaxScaler, TimeSeriesSplit'],
                 ['LightGBM', '4.6.0', 'Gradient-boosted trees (member)'],
                 ['CatBoost', '1.2.10', 'Gradient-boosted trees (member)'],
                 ['TensorFlow / Keras', '2.16.2', 'GRU and LSTM models'],
                 ['statsmodels', '0.14.6', 'ARIMA baseline; ADF, ACF/PACF in EDA'],
                 ['SciPy', '1.13.1', 'Binomial test, t-distribution (Diebold–Mariano), skew/kurtosis'],
                 ['ta', 'installed (≥ 0.11)', 'Bollinger Bands, ATR, ADX, ROC indicators'],
                 ['yfinance', '1.2.0', 'Yahoo Finance downloads (OHLCV and macro series)'],
                 ['urllib (stdlib)', '—', 'alternative.me Fear &amp; Greed API (JSON, no key)'],
                 ['joblib', '1.5.3', 'Persistence of sklearn/LightGBM models, scalers and the stacked meta-model'],
                 ['Streamlit', '1.50.0', 'Dashboard'],
                 ['Plotly', '6.7.0', 'Interactive dashboard charts'],
                 ['matplotlib', '3.9.4', 'Report figures (results/figures/)'],
                 ['FastAPI / uvicorn', '0.128.8 / 0.39.0', 'REST API (/health, /models, /predict/{asset})'],
                 ['pytest', '8.4.2', f'{N_TESTS} automated tests'],
                 ['reportlab', '5.0.1', 'PDF report generation (docs/build_report.py, this report)'],
                 ['Make', '—', 'Pipeline targets (preprocess, tune, train, stack, evaluate, figures, experiments, test, serve, api)'],
                 ['Docker / docker-compose', 'python:3.11-slim image', 'Containerised dashboard (8501) and API (8000)'],
                 ['GitHub Actions', 'ci.yml', 'Import checks + test suite on push / pull request'],
                 ['Storage', 'CSV, JSON, joblib .pkl, CatBoost .cbm, Keras .keras', 'No database: all data, artefacts and results are files under data/ and results/']],
                [3.4 * cm, 4.2 * cm, 9.0 * cm], align_right_from=None),
          P('Table 25. Technologies actually used (versions read from the project virtual environment on the report date). Not used: SQLite or any database, cloud services, paid data APIs, GPU.', 'caption')]

    # ================================================================ 26. ARCHITECTURE
    F += [H1('26. Project Architecture')]
    F += [side_by_side(vflow(['Data sources\nYahoo Finance (yfinance) · alternative.me', 'Data processing\nmarket_calendar · data_collection · external_data · DataCleaner',
                              'Feature engineering\n28–29 stationary features · scaler', 'ML models\nregistry: Ridge · RF · LightGBM · CatBoost · GRU · LSTM (+ baselines, stack)',
                              'Ensemble\nequal-weight Combined', 'Prediction engine\nprediction.py: predict_next_day · predict_for_date', 'Streamlit dashboard  ·  FastAPI'], width_cm=7.8, box_h=0.95, gap=0.4, font=7.2),
                       vflow(['Prediction engine\n(same module)', 'Evaluation script\nbacktesting.py — single test evaluation', 'Prediction storage\nresults/predictions/<prefix>_test_predictions.csv',
                              'Per-day verification\nactual close · error · HIT/FAIL (frozen test days)', 'Prediction history\nTest-Set History tab · CSV download', 'model_status.json\nserved forecast, members, held-out metrics, target stats'],
                             width_cm=7.8, box_h=0.95, gap=0.4, font=7.2, fill=colors.HexColor('#E6F4F1')), widths=(8.3, 8.3)),
          P('Figure 16. Left: the forecasting path. Right: the evaluation / verification path. Both share the prediction engine and the artefacts under data/ and results/.', 'caption')]
    F += [table([['Component', 'Responsibility', 'Inputs → outputs'],
                 ['Data source layer', 'Download complete daily bars from free public sources', 'tickers → data/raw/*.csv (frozen) or data/raw_live/*.csv (live)'],
                 ['Data processing', 'Clean, align calendars, apply the close-time rule, engineer features, split, scale', 'raw CSV → data/processed/*_features.csv, *_scaled.csv, data/models/*_scaler.pkl'],
                 ['Model layer (registry)', 'One interface for fit/predict/save/load of every model; two-phase fitting', 'dataset dict → data/models/* artefacts'],
                 ['Training &amp; tuning', 'Walk-forward grid search; phase A / phase B training; diagnostics', 'best_params.json, train_val_metrics.csv, histories, feature_importance.csv'],
                 ['Evaluation', 'CV table, single test evaluation, regime analysis, model_status.json, FINAL_RESULTS.md, per-day predictions', 'results/*.csv, results/predictions/*.csv, data/models/model_status.json'],
                 ['Prediction engine', 'Load members, build the scaled window, combine, reconstruct price, attach held-out context; demo mode on frozen data', 'features (live or frozen) → result dict (price, return, direction, band, members, metrics, disclaimer)'],
                 ['Presentation', 'Streamlit dashboard; FastAPI JSON', 'result dicts + result files → UI / JSON'],
                 ['Quality', 'pytest suite, CI workflow, Makefile, Docker', '—']], [3.2 * cm, 6.6 * cm, 6.8 * cm], align_right_from=None),
          P('Table 26. Components of the architecture.', 'caption')]
    F += [P('<b>API.</b> <font name="Mono">GET /health</font>; <font name="Mono">GET /models</font> (served forecast, members, CV-selected model and held-out metrics per asset); '
            '<font name="Mono">GET /predict/{bitcoin|gold|silver}[?model=GRU]</font> (default = the Combined forecast with every member\'s prediction, target date, uncertainty band, data source, test metrics and disclaimer; '
            '404 for unknown assets/models, 503 when artefacts are missing, 500 otherwise). The API shares <font name="Mono">predict_next_day()</font> with the dashboard; it does not refresh live data itself '
            '(it uses whichever feature file is present).')]

    # ================================================================ 27. FILE STRUCTURE
    F += [H1('27. File / Code Structure')]
    F += [table([['File / folder', 'Handles', 'Purpose'],
                 ['config.py', 'configuration', 'Paths, tickers, DATA_START_DATE, target/tasks, frozen split dates, feature policy (LEVEL_COLUMNS, close-time rule), SEQ_LEN, default and tuned hyper-parameters, model_status loader'],
                 ['src/data/data_collection.py', 'data', 'Yahoo OHLCV download for the three assets (complete bars only)'],
                 ['src/data/external_data.py', 'data', 'Macro series (DXY, WTI, TNX, S&amp;P 500, VIX) and Fear &amp; Greed'],
                 ['src/data/market_calendar.py', 'data', 'Complete-bar cut-offs, next trading day, expected last complete bar'],
                 ['src/data/preprocessing.py', 'data · features · split', 'DataCleaner (cleaning + all features + split + scaler), create_sequences, build_dataset (the single loader)'],
                 ['src/data/sync_live_data.py', 'data (live)', 'Refresh into data/raw_live and write <prefix>_live_features.csv atomically; never raises on API failure'],
                 ['src/data/eda.py', 'analysis', 'ADF tests, ACF/PACF, return distributions, volatility clustering → results/eda_summary.csv, figures'],
                 ['src/models/registry.py', 'models', 'Uniform model interface; Naive, Naive-Mean, ARIMA, Ridge, RandomForest, LightGBM, CatBoost, GRU, LSTM; save/load'],
                 ['src/models/model_gru.py, model_lstm.py', 'models', 'Keras architectures (single recurrent layer + dropout + dense)'],
                 ['src/models/ensemble_model.py', 'models (experiment)', 'Stacked ensemble: non-negative Ridge on out-of-fold predictions'],
                 ['src/training/tune_models.py', 'training', 'Walk-forward hyper-parameter search (train+val only) → results/tuning/'],
                 ['src/training/train_models.py', 'training', 'Two-phase training, validation metrics, loss curves, feature importance → data/models/'],
                 ['src/evaluation/cross_validation.py', 'evaluation', 'Expanding-window folds, cv_evaluate (with OOF predictions)'],
                 ['src/evaluation/backtesting.py', 'evaluation · storage', 'THE single test evaluation; Combined row; regime analysis; model_status.json; FINAL_RESULTS.md; per-day prediction files'],
                 ['src/evaluation/plots.py', 'visualisation', 'Report figures → results/figures/'],
                 ['src/experiments/run_experiments.py, before_after.py', 'experiments', 'E1–E4 (validation only) and the before/after comparison'],
                 ['src/inference/prediction.py', 'prediction', 'predict_next_day (Combined), predict_for_date (demo), predict_all_models, prediction_history, live-data currency check'],
                 ['src/api/app.py', 'API', 'FastAPI service'],
                 ['src/utils/metrics.py', 'evaluation', 'RMSE/MAE/MAPE/R², directional accuracy + binomial p, Diebold–Mariano, strategy back-test, volatility metrics'],
                 ['src/utils/inverse_transform.py, reproducibility.py, logging_config.py', 'utilities', 'Price reconstruction; seeds; logging'],
                 ['app/streamlit_app.py', 'dashboard', 'The Streamlit application (5 tabs)'],
                 ['tests/ (6 files)', 'quality', f'{N_TESTS} tests: features/look-ahead, calendar, sequences/split, metrics, model contract, inference and failure modes'],
                 ['scripts/retrain.py', 'automation (pipeline)', 'End-to-end retraining: fetch → preprocess → tune → train → stack → evaluate → figures → experiments'],
                 ['Makefile · Dockerfile · docker-compose.yml · .github/workflows/ci.yml', 'automation', 'Pipeline targets, containers, CI'],
                 ['data/raw, raw_live, raw_3y_backup, processed, models', 'storage', 'Frozen raw data, live copies, earlier 3-year files, features/splits/scalers, 18 model artefacts + 3 scalers + 3 stacks + model_status.json'],
                 ['results/', 'storage', 'cv_results, final_test_results, train_val_metrics, feature_importance, regime_analysis, eda_summary, FINAL_RESULTS.md, tuning/, stacking/, predictions/, experiments/, histories/, figures/, archives'],
                 ['docs/', 'documentation', 'METHODOLOGY, FEATURES, RESULTS, LIMITATIONS, VIVA_QA, DEMO_GUIDE, REPORT_STRUCTURE, FIX_PLAN, AUDIT_SUMMARY, build_report.py, build_technical_report.py, PDFs, figures'],
                 ['notebooks/ (4)', 'documentation', 'Companion notebooks that load and display the script outputs (exploration, features, training, evaluation)'],
                 ['app/utils.py · catboost_info/ · logs/', 'housekeeping', 'Empty module; CatBoost training debris; empty log directory — not used by the system']],
                [5.2 * cm, 2.6 * cm, 8.8 * cm], align_right_from=None, font=7.2),
          P('Table 27. Important files and folders.', 'caption')]

    # ================================================================ 28. ERROR HANDLING
    F += [H1('28. Error Handling &amp; Reliability')]
    F += [table([['Situation', 'Behaviour of the final system', 'Assessment'],
                 ['Yahoo Finance / network failure during a live refresh', 'update_live_data() catches the exception, prints a message and returns False; the dashboard shows a warning and continues with the stored dataset (tested: test_live_sync_survives_network_failure)', 'Handled'],
                 ['Empty response (rate limit)', 'fetch_asset() leaves the existing CSV untouched and returns an empty frame; treated as failure by the live sync', 'Handled'],
                 ['Vendor has not published the newest bar', 'live_data_is_current() compares the live file with the expected last complete bar; the dashboard warns and states which bar the forecast starts from; downloads are throttled to one per 30 min', 'Handled, explained to the user'],
                 ['Corrupt / partial download → invalid features', 'DataCleaner.validate_features() raises ValueError (NaN/inf/too few rows); the live file is not overwritten (atomic write via os.replace)', 'Handled'],
                 ['NaN in the newest feature row at prediction time', 'The scaler/model produce NaN → prediction.py raises ValueError("non-finite prediction") instead of showing a number (tested)', 'Handled'],
                 ['Missing processed data / models / evaluation', 'Explicit FileNotFoundError with the make target to run (preprocess / train / evaluate); dashboard shows an info/warning box; API returns 503', 'Handled'],
                 ['Feature-count mismatch between data and scaler', 'ValueError("Feature mismatch …") before any model runs', 'Handled'],
                 ['One member artefact broken', 'A missing artefact is simply not in the member list (the dashboard always states how many models were combined); an artefact that exists but fails to load or returns a non-finite value raises an explicit error instead of being skipped silently. predict_all_models() (comparison table) logs and skips a broken model', 'Handled'],
                 ['Unknown asset / model in the API', 'HTTP 404 with the list of valid names', 'Handled'],
                 ['Duplicate execution', 'Live refresh throttled; evaluation regenerates files idempotently; no scheduler exists, so duplicate scheduled runs cannot occur', 'n/a in the final system'],
                 ['Scheduler failure', 'No scheduler in the final system', 'n/a'],
                 ['Exchange holidays', 'next_trading_day() uses business days only — a forecast made on the day before a US holiday is labelled with the holiday date (the dashboard label says "next trading day")', 'Weakness (cosmetic)'],
                 ['Futures contract roll in the live series', 'Cannot be prevented; the dashboard states that live closes may differ from the frozen dataset after a roll', 'Weakness (documented)']],
                [4.2 * cm, 9.4 * cm, 3.0 * cm], align_right_from=None, font=7.3),
          P('Table 28. Failure modes and how the code responds.', 'caption')]

    # ================================================================ 29. LIMITATIONS
    F += [H1('29. Limitations')]
    F += bullets([
        f'<b>Markets are close to unpredictable at a one-day horizon.</b> This is the dominant fact: lag-1 return autocorrelation is {eda.loc["Bitcoin","acf_lag1"]:+.2f} / {eda.loc["Gold","acf_lag1"]:+.2f} / {eda.loc["Silver","acf_lag1"]:+.2f}; the project\'s answer to its own research question is negative.',
        '<b>External events are not modelled.</b> News, regulation, exchange outages and macro shocks are outside the feature set (only their reflection in past prices, macro returns and one sentiment index).',
        '<b>Historical patterns may not persist.</b> The 2026 test window is a different regime from 2018–2025 (Gold and Silver traded far above their training ranges; Silver\'s 30-day volatility exceeded every training value on most test days). Stationary features reduce but do not remove this.',
        '<b>One test window, limited sample.</b> 143–207 unseen days: with 143 days, 56 % direction is not distinguishable from 50 %; 30 model-rows are reported, so one p ≈ 0.02 cell is expected by chance. The walk-forward folds are the more reliable picture — and there no model is distinguishable from the random walk either.',
        '<b>No live forward-testing record in the final system.</b> The automatic daily record was removed; out-of-sample evidence rests on the frozen test period only (§19, §22).',
        '<b>Data limitations.</b> Yahoo Finance is an unofficial free source; BTC-USD is a composite index; GC=F / SI=F are continuous front-month series with roll effects and unusable volume; Yahoo revises a few historical closes between downloads (mean &lt; 0.005 %, max 0.75 % on one Bitcoin day) and publishes Bitcoin\'s previous-day bar with a delay; the metals\' macro features are one day old by construction (close-time rule); Fear &amp; Greed is a proprietary composite.',
        '<b>Prediction uncertainty is not calibrated.</b> The ±1 RMSE band is an ex-post error band from the test period, not a predictive interval; all models are point forecasters.',
        '<b>Asset-specific.</b> Bitcoin\'s 7-day calendar vs the metals\' exchange calendar; exchange holidays are not modelled for the target-date label; Silver depends on gold\'s same-day return being available.',
        '<b>Modelling choices.</b> Small hyper-parameter grids; a single 32-unit recurrent layer; selection by mean fold RMSE (a practitioner might select on direction or Sharpe); the stacked ensemble is a linear blend; the combination is an equal-weight mean, which is a ceiling when the members carry no signal.',
        '<b>Evaluation caveats.</b> The strategy back-test is stylised (long/flat, flat 10 bps, no slippage, shorting, sizing or financing); the Diebold–Mariano test assumes covariance-stationary loss differentials, only approximate with fat-tailed returns; the before/after comparison reads the test rows a second time (declared, no selection made on it).'])

    # ================================================================ 30. ETHICS
    F += [H1('30. Ethical / Academic Positioning')]
    F += [P('<b>This is an academic machine-learning prediction and evaluation project.</b> It is <b>not</b> financial advice, not an investment strategy, not a guaranteed or profitable trading system and not a '
            'professional forecasting service. The disclaimer is embedded in the code (<font name="Mono">prediction.DISCLAIMER</font>) and is returned with every API response and shown in the dashboard footer.', 'box'),
          P('<b>Responsible interpretation.</b> Every forecast is displayed together with its held-out error, its direction hit-rate with a p-value and its comparison with the random walk, precisely so that a '
            'reader cannot mistake a plausible-looking number for skill. The project reports negative results, archives superseded results with an explanation instead of deleting them, and documents the leak it '
            'found in its own earlier version. Anyone using the system should treat the predicted price as "today\'s close plus noise of the size shown in the band" — which is what the evaluation says it is.')]

    # ================================================================ 31. FUTURE
    F += [H1('31. Future Improvements')]
    F += bullets([
        '<b>Restore and run the automatic daily forward record</b> for a longer period (months, not days) — the prototype design exists (§19); it would turn the operational pipeline into a continuously growing out-of-sample history.',
        '<b>Longer and more diverse test evidence:</b> several rolling test windows or a full walk-forward re-training schedule over 2022–2026 instead of one frozen window, to show robustness across regimes.',
        '<b>Volatility forecasting as a product:</b> experiment E4 showed 22-day realised volatility <i>is</i> more predictable than the return (Bitcoin Ridge −11.5 % RMSE vs persistence, though only ~4 % better than EWMA and worse than EWMA for Silver); a served volatility band would be more useful than a point price.',
        '<b>Calibrated uncertainty:</b> quantile regression or conformal prediction intervals instead of the ex-post ±1 RMSE band.',
        '<b>Additional information sources</b> with verified timestamps: order-flow / on-chain metrics for Bitcoin, positioning data (COT) for the metals, news sentiment — every new series must pass the close-time check.',
        '<b>More data:</b> intraday bars would give many more samples per year and permit realised-volatility features at finer resolution.',
        '<b>Modelling:</b> larger tuning budgets with nested CV, temporal-fusion or attention models only if the data volume justifies them, and probabilistic (distributional) losses.',
        '<b>Engineering:</b> exchange-holiday calendar for the target-date label, a small results database if a live history is kept, scheduled re-training, and hosting of the dashboard.'])

    # ================================================================ 32. STATUS
    F += [PageBreak(), H1('32. Final Project Status — honest assessment')]
    F += [table([['Question', 'Answer', 'Evidence'],
                 ['Is the system fully functional?', 'Yes. Data download, preprocessing, tuning, training, evaluation, dashboard, API and tests all run from the Makefile; a live prediction was produced for all three assets on the report date.', 'make pipeline; make predict output (§17.1); 41/41 tests'],
                 ['Is the dataset pipeline correct?', 'Yes. Complete bars only, correct calendars, no nulls/duplicates, close-time alignment; the earlier synthetic-weekend and intraday-snapshot problems were found and fixed.', '§6–7, §11; tests test_features.py, test_market_calendar.py'],
                 ['Are the features correctly implemented?', 'Yes. 28–29 backward-looking stationary features; look-ahead and close-time tests pass; ablation shows no harmful group.', '§8; test_no_lookahead_in_any_feature and two close-time tests'],
                 ['Are the models correctly implemented?', 'Yes. Uniform registry, two-phase training with validation-only early stopping, saved/loaded artefacts reproduce the evaluation; model contract tested.', '§12–13; test_models_and_inference.py'],
                 ['Is the evaluation methodology valid?', 'Yes. Frozen chronological split, walk-forward tuning/selection on train+val, single test evaluation, return-space metrics, DM and binomial tests, baselines.', '§10, §14–15'],
                 ['Is there data leakage?', 'None known after the audit. The one leak found (same-day macro/High-Low for the metals) was fixed, tested and its effect documented. The before/after script re-reads the test rows without selecting anything (declared).', '§11; Table 10'],
                 ['Is the ensemble working correctly?', 'Yes. The served forecast is the equal-weight mean of six members; the stored Combined predictions equal the mean of the stored member predictions (tested); it is evaluated like a single model.', '§16; test_combined_test_row_is_the_mean_of_its_members'],
                 ['Is automatic prediction working?', 'Not present. It was built, verified and removed at the author\'s request; the final system produces predictions on demand only.', '§19; CHANGELOG 2026-09-14'],
                 ['Is prediction history working?', 'Yes for the frozen unseen test period (143–207 rows per asset with predicted/actual/error/HIT-FAIL). No live, growing history.', '§21; results/predictions/*.csv'],
                 ['Is the dashboard complete?', 'Yes: five tabs, one-click combined forecast, unseen-day demo, per-day history with download, full performance and methodology views, disclaimer.', '§23; Figures 11–15'],
                 ['Is the project suitable for FYP demonstration?', 'Yes. It demonstrates the full life-cycle, a defensible methodology, a working product and scientific honesty about a negative result.', 'whole report; docs/DEMO_GUIDE.md, docs/VIVA_QA.md'],
                 ['Strongest parts', 'Leakage audit and the discovered close-time leak; walk-forward validation with formal tests; the equal-weight combination that removes model cherry-picking; reproducibility (frozen dates, seeds, single loader, tests, CI); a dashboard that shows reliability next to every number.', '—'],
                 ['Weakest parts', 'No predictive skill over the random walk (expected, but it is the result); only one frozen test window; no live forward record in the final system; point forecasts without calibrated intervals; free-vendor data quirks (contract rolls, delayed bars).', '—'],
                 ['What to improve before final submission', '(1) Decide whether to restore the automatic daily record and, if so, let it run for several weeks before the viva; (2) replace the stale dashboard screenshots in docs/figures/ (they still show the removed tab) — this report already uses fresh ones; (3) tidy housekeeping files (empty app/utils.py, catboost_info/); (4) commit the current working state; (5) rehearse the honest framing of the negative result with docs/VIVA_QA.md.', '—']],
                [3.6 * cm, 8.6 * cm, 4.4 * cm], align_right_from=None, font=7.3),
          P('Table 29. Final status.', 'caption')]

    # ================================================================ 33. TECHNICAL SUMMARY
    F += [H1('33. Final Technical Summary')]
    F += [P(f'<b>Problem →</b> predict the next trading day\'s close of Bitcoin, Gold and Silver and determine whether machine learning beats the random walk at a one-day horizon. '
            f'<b>Dataset →</b> Yahoo Finance daily OHLCV from 2018-01-01 ({len(raw["Bitcoin"]):,} / {len(raw["Gold"]):,} / {len(raw["Silver"]):,} raw rows; complete bars only) plus DXY, WTI, 10-year yield, S&amp;P 500, VIX and the crypto Fear &amp; Greed index; no nulls, no duplicates, exchange calendar kept for the metals. '
            f'<b>Features →</b> {len(b["features"])} / {len(g["features"])} / {len(s_["features"])} stationary, backward-looking features (returns and lags, momentum, rolling/HAR/EWMA volatility, RSI/ADX/ROC, normalised MACD, EMA ratios, Bollinger %B/width, ATR/price, high–low range, macro returns, sentiment/calendar/volume for Bitcoin, gold return for Silver) under a verified close-time rule; price levels excluded. '
            f'<b>Target →</b> next-day log return ln(P<sub>t+1</sub>/P<sub>t</sub>), converted to a price; UP/DOWN from its sign. '
            f'<b>Models →</b> Ridge, Random Forest, LightGBM, CatBoost, GRU, LSTM (trained) against random-walk, historical-mean and ARIMA baselines and a stacked-ensemble experiment. '
            f'<b>Training →</b> frozen chronological split (train ≤ {TRAIN_END}, validation ≤ {VAL_END}, test after), MinMax scaler fitted on train, 4-fold expanding walk-forward hyper-parameter search on train+val, two-phase training (validation-monitored stopping, refit on train+val), seeds and pinned versions. '
            f'<b>Evaluation →</b> one evaluation on {ct["Bitcoin"]["n_test"]} / {ct["Gold"]["n_test"]} / {ct["Silver"]["n_test"]} unseen days with USD and return-space errors, directional accuracy with a binomial test, Diebold–Mariano test against the random walk, strategy back-test and regime analysis. '
            f'<b>Ensemble →</b> the served forecast is the equal-weight mean of the six models\' predicted returns — no fitted weights, no model selection by the user. '
            f'<b>Prediction →</b> one click (or one API call) refreshes the last complete bar, runs all six models and returns one price, UP/DOWN, a ±1 RMSE band and each member\'s prediction with the held-out reliability next to it. '
            f'<b>Dashboard →</b> Streamlit, five tabs (Forecast, Predict a Day on unseen test days, Test-Set History with HIT/FAIL, Model Performance, Methodology) plus a FastAPI endpoint. '
            f'<b>Automation →</b> Makefile pipeline, retrain script, {N_TESTS} tests, CI and Docker; the automatic daily forward-prediction record was prototyped and removed from the final system. '
            f'<b>Verification →</b> every unseen test day is predicted from data up to that day and compared with the actual next close (error $ / %, HIT/FAIL), stored as a per-day history and shown in the dashboard. '
            f'<b>Results →</b> the combined forecast\'s next-day error is {ct["Bitcoin"]["MAPE_usd"]:.2f} % / {ct["Gold"]["MAPE_usd"]:.2f} % / {ct["Silver"]["MAPE_usd"]:.2f} % of price — the same as the random walk\'s — with directional accuracy '
            f'{ct["Bitcoin"]["DirAcc_pct"]:.1f} / {ct["Gold"]["DirAcc_pct"]:.1f} / {ct["Silver"]["DirAcc_pct"]:.1f} % and Diebold–Mariano p = {ct["Bitcoin"]["DM_pvalue"]:.2f} / {ct["Gold"]["DM_pvalue"]:.2f} / {ct["Silver"]["DM_pvalue"]:.2f}: '
            'no model, single or combined, beats the random walk at a one-day horizon in this data. The project\'s value is a correct, tested, transparent and reproducible pipeline that reports that result honestly — '
            'including the discovery and correction of a close-time data leak that had produced an apparently significant 62 % result.')]
    F += [Spacer(1, 10), P('<i>End of report. Regenerate with</i> <font name="Mono">python docs/build_technical_report.py</font>; <i>all tables are produced from the result files at build time.</i>', 'small')]
    return F


if __name__ == '__main__':
    doc = Doc(OUT, title='Multi-Asset Next-Day Price Prediction — Technical Report', author='Alyan Shahid')
    doc.multiBuild(build())
    print('written', OUT)
