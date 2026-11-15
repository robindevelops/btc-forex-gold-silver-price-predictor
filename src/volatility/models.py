"""
Volatility models (all use only information available at day t).

    HAR (log form, Corsi 2009)   ln σ̂ = b0 + b1·ln rv_1d* + b2·ln rv_5d + b3·ln rv_22d   [+ b4·ln IV_t  → HAR-IV]
    HAR-1d (variance form)       r̂²_{t+1} = b0 + b1·rv_1d² + b2·rv_5d² + b3·rv_22d²      → σ_t = sqrt(max(r̂², floor))
    GARCH(1,1)                   σ²_{t+1} = ω + α·r_t² + β·σ²_t, Gaussian maximum likelihood on training returns

rv_1d, rv_5d, rv_22d and ewma_vol are the backward-looking volatility features built in src/data/preprocessing.py.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

HAR_COLS = ['rv_1d', 'rv_5d', 'rv_22d']


# ---------------------------------------------------------------- linear (OLS) volatility regressions
def ols_fit(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, np.ravel(y), rcond=None)
    return coef


def ols_predict(coef, X):
    return coef[0] + np.asarray(X) @ coef[1:]


def har_log_design(feats, dates, rv1_floor, iv=None):
    """ln rv_1d* (floored: a zero daily return has ln 0 = −∞), ln rv_5d, ln rv_22d [, ln IV] at each day t."""
    f = feats.loc[pd.DatetimeIndex(dates), HAR_COLS]
    cols = [np.log(np.maximum(f['rv_1d'].values, rv1_floor)), np.log(np.maximum(f['rv_5d'].values, 1e-6)),
            np.log(np.maximum(f['rv_22d'].values, 1e-6))]
    if iv is not None:
        cols.append(np.log(np.asarray(iv, float)))
    return np.column_stack(cols)


MIN_FIT_ROWS = 250        # ~1 year: a fit on fewer complete rows (e.g. an early walk-forward fold before DVOL exists) is not a model


class HARLog:
    """Log-HAR(-IV) forecast of the log realised volatility target."""

    def __init__(self, with_iv=False):
        self.with_iv = with_iv
        self.coef = None
        self.rv1_floor = None

    def fit(self, feats, dates, y, iv=None):
        self.rv1_floor = float(np.quantile(feats.loc[pd.DatetimeIndex(dates), 'rv_1d'].values, 0.05))
        self.rv1_floor = max(self.rv1_floor, 1e-5)
        X = har_log_design(feats, dates, self.rv1_floor, iv if self.with_iv else None)
        ok = np.isfinite(X).all(1) & np.isfinite(np.ravel(y))
        self.n_fit = int(ok.sum())
        self.coef = ols_fit(X[ok], np.ravel(y)[ok]) if self.n_fit >= MIN_FIT_ROWS else None
        return self

    def predict(self, feats, dates, iv=None):
        X = har_log_design(feats, dates, self.rv1_floor, iv if self.with_iv else None)
        out = np.full(len(X), np.nan)
        if self.coef is None:                     # too few rows to fit (see MIN_FIT_ROWS): no forecast, not a zero forecast
            return out
        ok = np.isfinite(X).all(1)
        out[ok] = ols_predict(self.coef, X[ok])
        return out

    def to_dict(self):
        return {'with_iv': self.with_iv, 'coef': [float(c) for c in self.coef], 'rv1_floor': self.rv1_floor,
                'inputs': ['1'] + [f'ln {c}' for c in HAR_COLS] + (['ln IV'] if self.with_iv else [])}

    @classmethod
    def from_dict(cls, d):
        m = cls(d['with_iv'])
        m.coef, m.rv1_floor = np.array(d['coef']), d['rv1_floor']
        return m


class HAR1d:
    """One-day volatility σ_t for the band: OLS of r²_{t+1} on squared HAR volatilities, floored."""

    def fit(self, feats, dates, next_ret):
        X = feats.loc[pd.DatetimeIndex(dates), HAR_COLS].values ** 2
        y = np.ravel(next_ret) ** 2
        self.coef = ols_fit(X, y)
        self.floor = float(max(np.quantile(y, 0.01), 1e-8))
        return self

    def sigma(self, feats, dates):
        X = feats.loc[pd.DatetimeIndex(dates), HAR_COLS].values ** 2
        return np.sqrt(np.maximum(ols_predict(self.coef, X), self.floor))

    def to_dict(self):
        return {'coef': [float(c) for c in self.coef], 'floor': self.floor, 'inputs': ['1'] + [f'{c}²' for c in HAR_COLS]}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.coef, m.floor = np.array(d['coef']), d['floor']
        return m


# ---------------------------------------------------------------- GARCH(1,1)
def _garch_var(r, omega, alpha, beta, var0):
    """σ²_{t+1|t} for every t: element i is the variance forecast for the return AFTER r[i]."""
    out = np.empty(len(r))
    v = var0
    for i, x in enumerate(r):
        v = omega + alpha * x * x + beta * v
        out[i] = v
    return out


class GARCH11:
    """Gaussian GARCH(1,1) on percent returns (numerically stabler); fitted once, then filtered with fixed parameters."""

    def fit(self, returns):
        r = np.asarray(returns, float) * 100
        var0 = float(np.var(r))

        def nll(p):
            omega, alpha, beta = p
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
                return 1e12
            v = np.concatenate([[var0], _garch_var(r[:-1], omega, alpha, beta, var0)])   # variance of r[i] given r[:i]
            return 0.5 * np.sum(np.log(v) + r * r / v)

        best = None
        for a0, b0 in ((0.05, 0.90), (0.10, 0.85), (0.03, 0.95)):
            res = minimize(nll, [var0 * (1 - a0 - b0), a0, b0], method='Nelder-Mead',
                           options={'maxiter': 4000, 'xatol': 1e-8, 'fatol': 1e-8})
            if best is None or res.fun < best.fun:
                best = res
        self.omega, self.alpha, self.beta = map(float, best.x)
        self.var0 = var0
        self.converged = bool(best.success)
        return self

    def sigma(self, returns):
        """σ_{t+1|t} (decimal) for each element of `returns` (the forecast made at that day's close)."""
        r = np.asarray(returns, float) * 100
        return np.sqrt(_garch_var(r, self.omega, self.alpha, self.beta, self.var0)) / 100

    def to_dict(self):
        return {'omega_pct2': self.omega, 'alpha': self.alpha, 'beta': self.beta, 'var0_pct2': self.var0,
                'persistence': self.alpha + self.beta, 'converged': self.converged}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.omega, m.alpha, m.beta, m.var0 = d['omega_pct2'], d['alpha'], d['beta'], d['var0_pct2']
        m.converged = d.get('converged', True)
        return m
