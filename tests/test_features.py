"""Feature-engineering guarantees: correctness of indicators and NO look-ahead."""
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


def test_no_lookahead_in_any_feature(synthetic_price_data):
    """Changing the LAST observation must not change any feature at an earlier date."""
    a = _cleaner(synthetic_price_data); a.clean_data()
    df2 = synthetic_price_data.copy()
    df2.loc[df2.index[-1], ['open', 'high', 'low', 'price', 'volume']] *= [1.5, 1.7, 1.2, 1.5, 3.0]
    b = _cleaner(df2); b.clean_data()
    common = a.df.index[:-1]
    pd.testing.assert_frame_equal(a.df.loc[common], b.df.loc[common], check_exact=False, rtol=1e-9)


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
