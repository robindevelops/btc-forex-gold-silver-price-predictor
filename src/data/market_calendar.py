"""
Complete-bar rule and trading-calendar helpers shared by data collection, live sync and inference.

Yahoo Finance returns the CURRENT day's row while the day is still trading, as if it were a finished bar.
A model that sees that row is looking at an intraday snapshot, and a "prediction" verified against it is not
verified against a close. Every download therefore drops rows whose bar is not yet complete:

    crypto (BTC-USD)      Yahoo's daily bar is the UTC day → complete once the next UTC day has started.
    commodity (GC=F/SI=F) the daily row is the ET trading day; the close is the 13:30 ET settlement but Yahoo
                          keeps revising the row (High/Low, volume) until the Globex session ends at 17:00 ET
                          → treated as complete from 17:15 ET.
    US market series      (S&P 500, VIX, 10-y yield, DXY, WTI) close between 14:30 and 17:00 ET → same rule
                          as the commodities.
"""
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo('America/New_York')
SESSION_END_ET = (17, 15)          # (hour, minute) after which today's futures / US-market row is final


def complete_bar_cutoff(asset_type, now=None):
    """
    First calendar date whose daily bar is NOT yet complete. Rows dated on or after it must be dropped.
        crypto     → today's UTC date
        commodity  → today's ET date before 17:15 ET, tomorrow's ET date after
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    if asset_type == 'crypto':
        return now.astimezone(dt.timezone.utc).date()
    et = now.astimezone(ET)
    return et.date() + dt.timedelta(days=1) if (et.hour, et.minute) >= SESSION_END_ET else et.date()


def drop_incomplete_bars(df, asset_type, now=None, date_col='timestamp'):
    """Return only the rows whose bar is complete (see `complete_bar_cutoff`)."""
    cutoff = complete_bar_cutoff(asset_type, now)
    dates = pd.to_datetime(df[date_col]).dt.date
    return df[dates < cutoff].copy()


def next_trading_day(asset_type, date):
    """Expected date of the next bar: the next calendar day for crypto, the next weekday for exchange-traded
    futures (exchange holidays are not modelled — the label on the dashboard says "next trading day")."""
    d = pd.Timestamp(date)
    if asset_type == 'crypto':
        return (d + pd.Timedelta(days=1)).date()
    return (d + pd.tseries.offsets.BDay(1)).date()


def expected_last_complete_bar(asset_type, now=None):
    """Date of the newest bar that should exist right now (ignoring exchange holidays)."""
    cutoff = complete_bar_cutoff(asset_type, now)
    if asset_type == 'crypto':
        return cutoff - dt.timedelta(days=1)
    return (pd.Timestamp(cutoff) - pd.tseries.offsets.BDay(1)).date()
