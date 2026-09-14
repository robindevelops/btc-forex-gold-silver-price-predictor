"""Feature-engineering guarantees: correctness of indicators and NO look-ahead."""
import os
import numpy as np
import pandas as pd
import pytest
from src.data.preprocessing import DataCleaner
from config import LEVEL_COLUMNS


def _cleaner(df, asset='Bitcoin'):
    c = DataCleaner(asset)
    c.df = df.copy()
    return c


def test_full_clean_has_no_nans_and_only_stationary_model_features(synthetic_price_data):
    c = _cleaner(synthetic_price_data)
    c.clean_data()
    assert c.df.isna().sum().sum() == 0
    model_cols = c.model_feature_columns()
    assert not set(model_cols) & set(LEVEL_COLUMNS)
    assert 'log_return' in model_cols and 'price' not in model_cols


def test_log_return_definition(synthetic_price_data):
    c = _cleaner(synthetic_price_data); c.clean_data()
    p = c.df['price']
    expected = np.log(p / p.shift(1)).dropna()
    assert np.allclose(c.df['log_return'].iloc[1:], expected)


def test_rsi_bounds_and_bollinger_ordering(synthetic_price_data):
    c = _cleaner(synthetic_price_data); c.clean_data()
    assert c.df['RSI'].between(0, 100).all()
    assert (c.df['BB_Upper'] >= c.df['BB_Mid']).all() and (c.df['BB_Mid'] >= c.df['BB_Lower']).all()
    assert c.df['bb_pctb'].abs().max() < 5          # %B is ~[0,1] with occasional excursions


def test_lagged_returns_are_strictly_past(synthetic_price_data):
    c = _cleaner(synthetic_price_data); c.clean_data()
    lr = c.df['log_return']
    for k in (1, 2, 5, 10):
        assert np.allclose(c.df[f'return_{k}d'].iloc[k:], lr.shift(k).iloc[k:])


@pytest.mark.parametrize('asset', ['Bitcoin', 'Gold'])
def test_no_lookahead_in_any_feature(synthetic_price_data, asset):
    """Changing the LAST observation must not change any feature at an earlier date (crypto and futures paths)."""
    a = _cleaner(synthetic_price_data, asset); a.clean_data()
    df2 = synthetic_price_data.copy()
    df2.loc[df2.index[-1], ['open', 'high', 'low', 'price', 'volume']] *= [1.5, 1.7, 1.2, 1.5, 3.0]
    b = _cleaner(df2, asset); b.clean_data()
    common = a.df.index[:-1]
    pd.testing.assert_frame_equal(a.df.loc[common], b.df.loc[common], check_exact=False, rtol=1e-9)


def _macro_return(fname):
    from config import RAW_DATA_DIR
    s = pd.read_csv(os.path.join(RAW_DATA_DIR, fname), parse_dates=['timestamp']).set_index('timestamp')['price']
    s = s[~s.index.duplicated()].sort_index()
    return np.log(s / s.shift(1))


@pytest.mark.skipif(not os.path.exists(os.path.join(__import__('config').RAW_DATA_DIR, 'sp500_data.csv')), reason="raw macro data not present (CI)")
def test_commodity_external_features_are_previous_day_values(synthetic_price_data):
    """
    GC=F / SI=F close at the 13:30 ET settlement, BEFORE the S&P/VIX/DXY/TNX/WTI closes: at row t a metal
    may only see the macro return of the latest external date STRICTLY before t. Bitcoin (00:00 UTC close,
    after the US close) may see the same day's value.
    """
    sp = _macro_return('sp500_data.csv')
    g = _cleaner(synthetic_price_data, 'Gold'); g.clean_data()
    for t in g.df.index[::37]:
        prior = sp[sp.index < t]
        assert np.isclose(g.df.loc[t, 'sp500_return'], prior.iloc[-1]), t          # strictly earlier date
    same_day = [t for t in g.df.index if t in sp.index]
    assert not np.allclose(g.df.loc[same_day, 'sp500_return'], sp.loc[same_day])  # and NOT the same-day value
    b = _cleaner(synthetic_price_data, 'Bitcoin'); b.clean_data()
    for t in b.df.index[::37]:
        same_or_prior = sp[sp.index <= t]
        assert np.isclose(b.df.loc[t, 'sp500_return'], same_or_prior.iloc[-1]), t


def test_commodity_high_low_features_are_lagged_one_session(synthetic_price_data):
    """Yahoo's futures High/Low span the session past the settlement: hl_range/atr_norm/ADX are lagged for metals only."""
    from config import POST_SETTLEMENT_FEATURES
    g = _cleaner(synthetic_price_data, 'Gold'); g.clean_data()
    raw_hl = ((g.df['high'] - g.df['low']) / g.df['price'])
    assert np.allclose(g.df['hl_range'].iloc[1:], raw_hl.shift(1).iloc[1:])          # previous session's range
    assert not np.allclose(g.df['hl_range'], raw_hl)
    assert set(POST_SETTLEMENT_FEATURES) <= set(g.df.columns)
    b = _cleaner(synthetic_price_data, 'Bitcoin'); b.clean_data()
    assert np.allclose(b.df['hl_range'], (b.df['high'] - b.df['low']) / b.df['price'])   # same-day for crypto


def test_commodities_keep_trading_calendar():
    """Gold/Silver must NOT get synthetic weekend rows (log_return == 0 artefacts)."""
    n = 260
    dates = pd.bdate_range('2023-01-02', periods=n)          # business days only
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, n)))
    df = pd.DataFrame({'timestamp': dates, 'open': prices, 'high': prices * 1.01, 'low': prices * 0.99,
                       'price': prices, 'volume': 1000.0})
    c = _cleaner(df, 'Gold'); c.clean_data()
    assert (c.df.index.dayofweek < 5).all()
    assert (c.df['log_return'] == 0).sum() == 0
