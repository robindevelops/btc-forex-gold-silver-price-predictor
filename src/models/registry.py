"""
Model registry: one uniform interface for every model compared in the project.

    model = make_model(name, params)
    model.fit(data, train_mask_or_arrays...)      # via fit_on()
    yhat  = model.predict(Xseq, Xt)               # scaled log-return predictions, shape (n,)

Recurrent models consume the (n, seq_len, F) windows; tabular models consume the
last row of each window (n, F), i.e. today's indicators. Both predict the same
target, so all models are compared on identical samples.

Two-phase training for models that need a validation signal (epochs / iterations):
    phase 1: fit on train, monitor val  → pick the stopping point
    phase 2: refit on train+val with that stopping point fixed  (val never leaks into
             its own early-stopping monitor — this was the CatBoost bug in the old code)
"""
import os
import json
import numpy as np
import joblib

from config import MODELS_DIR, get_prefix
from src.utils.reproducibility import set_all_seeds

def artefact_stem(asset_name, model_name, task='return_1d'):
    """data/models/<prefix>_<model>[_<task>] — the served return task keeps the short name."""
    stem = f'{get_prefix(asset_name)}_{model_name.lower()}'
    return stem if task == 'return_1d' else f'{stem}_{task}'


TABULAR = ('Ridge', 'RandomForest', 'LightGBM', 'CatBoost')
RECURRENT = ('GRU', 'LSTM')
BASELINES = ('Naive', 'Naive-Mean', 'ARIMA')
ALL_MODELS = BASELINES + TABULAR + RECURRENT


class BaseModel:
    name = 'Base'
    kind = 'tabular'          # 'tabular' | 'recurrent' | 'baseline'

    def __init__(self, params=None):
        self.params = dict(params or {})
        self.model = None
        self.history = None   # keras training history (recurrent models)
        self.fit_info = {}    # e.g. chosen epochs / iterations

    def fit(self, Xseq_tr, Xt_tr, y_tr, Xseq_val=None, Xt_val=None, y_val=None, final=False):
        raise NotImplementedError

    def predict(self, Xseq, Xt):
        raise NotImplementedError

    def feature_importance(self, columns):
        return None

    # ----- persistence
    def save(self, asset_name, task='return_1d'):
        path = os.path.join(MODELS_DIR, f'{artefact_stem(asset_name, self.name, task)}.pkl')
        joblib.dump({'model': self.model, 'params': self.params, 'fit_info': self.fit_info}, path)
        return path

    def load(self, asset_name, task='return_1d'):
        path = os.path.join(MODELS_DIR, f'{artefact_stem(asset_name, self.name, task)}.pkl')
        obj = joblib.load(path)
        self.model, self.params, self.fit_info = obj['model'], obj['params'], obj['fit_info']
        return self


# ============================================================ baselines
class NaiveModel(BaseModel):
    """
    Task-specific naive forecast, THE baseline every model must beat:
        return tasks : r̂ = 0  (random walk: P̂_{t+h} = P_t)
        vol task     : σ̂ = last realised volatility (persistence)
    The actual naive values come from the dataset (`data['naive_<split>']`); this class only
    carries a constant for API compatibility.
    """
    name, kind = 'Naive', 'baseline'

    def __init__(self, params=None, constant=0.0):
        super().__init__(params)
        self.constant = constant

    def fit(self, Xseq_tr, Xt_tr, y_tr, *a, **k):
        return self

    def predict(self, Xseq, Xt):
        return np.full(len(Xt), self.constant)

    def save(self, asset_name, task='return_1d'):
        return None


class NaiveMean(BaseModel):
    """Historical-mean forecast: r̂ = mean training return (drift)."""
    name, kind = 'Naive-Mean', 'baseline'

    def fit(self, Xseq_tr, Xt_tr, y_tr, *a, **k):
        self.model = float(np.mean(y_tr))
        return self

    def predict(self, Xseq, Xt):
        return np.full(len(Xt), self.model)

    def save(self, asset_name, task='return_1d'):
        return None


class ARIMAModel(BaseModel):
    """
    ARIMA(p,0,q) on the standardised target series (returns are already the differenced series,
    so d=0). Order chosen by AIC on the training window. One-step-ahead forecasts are made
    walk-forward: after each day the observed value is appended (no refit).
    Needs the raw series, so it is handled specially in evaluation (see fit_series/forecast).
    """
    name, kind = 'ARIMA', 'baseline'

    def fit_series(self, y_train_series, max_p=3, max_q=3):
        import warnings
        from statsmodels.tsa.arima.model import ARIMA
        warnings.filterwarnings('ignore')
        best = (np.inf, None, None)
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    res = ARIMA(y_train_series, order=(p, 0, q)).fit()
                    if res.aic < best[0]:
                        best = (res.aic, (p, 0, q), res)
                except Exception:
                    continue
        self.fit_info = {'order': best[1], 'aic': float(best[0])}
        self.model = best[2]
        return self

    def forecast_walk_forward(self, y_future_series):
        """1-step-ahead forecasts for each value in y_future_series (appending truth as it arrives)."""
        res = self.model
        preds = []
        for v in np.ravel(y_future_series):
            preds.append(float(res.forecast(steps=1)[0]))
            res = res.append([v], refit=False)
        return np.array(preds)

    def save(self, asset_name, task='return_1d'):
        return None


# ============================================================ tabular
class RidgeModel(BaseModel):
    name = 'Ridge'

    def fit(self, Xseq_tr, Xt_tr, y_tr, Xseq_val=None, Xt_val=None, y_val=None, final=False):
        from sklearn.linear_model import Ridge
        X = np.concatenate([Xt_tr, Xt_val]) if final and Xt_val is not None else Xt_tr
        y = np.concatenate([y_tr, y_val]) if final and y_val is not None else y_tr
        self.model = Ridge(alpha=self.params.get('alpha', 10.0)).fit(X, np.ravel(y))
        return self

    def predict(self, Xseq, Xt):
        return self.model.predict(Xt)

    def feature_importance(self, columns):
        return dict(zip(columns, np.abs(self.model.coef_)))


class RandomForestModel(BaseModel):
    name = 'RandomForest'

    def fit(self, Xseq_tr, Xt_tr, y_tr, Xseq_val=None, Xt_val=None, y_val=None, final=False):
        from sklearn.ensemble import RandomForestRegressor
        X = np.concatenate([Xt_tr, Xt_val]) if final and Xt_val is not None else Xt_tr
        y = np.concatenate([y_tr, y_val]) if final and y_val is not None else y_tr
        self.model = RandomForestRegressor(random_state=42, n_jobs=-1, **self.params).fit(X, np.ravel(y))
        return self

    def predict(self, Xseq, Xt):
        return self.model.predict(Xt)

    def feature_importance(self, columns):
        return dict(zip(columns, self.model.feature_importances_))


class LightGBMModel(BaseModel):
    name = 'LightGBM'

    def fit(self, Xseq_tr, Xt_tr, y_tr, Xseq_val=None, Xt_val=None, y_val=None, final=False):
        import lightgbm as lgb
        params = dict(self.params)
        n_est = params.pop('n_estimators', 300)
        base = dict(random_state=42, n_jobs=-1, verbose=-1, **params)
        if Xt_val is not None and y_val is not None:
            # phase 1: early stopping on val to find the number of trees
            m1 = lgb.LGBMRegressor(n_estimators=n_est, **base)
            m1.fit(Xt_tr, np.ravel(y_tr), eval_set=[(Xt_val, np.ravel(y_val))],
                   callbacks=[lgb.early_stopping(30, verbose=False)])
            best_n = max(10, int(m1.best_iteration_ or n_est))
            self.fit_info = {'n_estimators_used': best_n}
            if not final:
                self.model = m1
                return self
            X = np.concatenate([Xt_tr, Xt_val]); y = np.concatenate([y_tr, y_val])
        else:
            best_n, X, y = n_est, Xt_tr, y_tr
            self.fit_info = {'n_estimators_used': best_n}
        # phase 2: refit on train+val with fixed tree count
        self.model = lgb.LGBMRegressor(n_estimators=best_n, **base).fit(X, np.ravel(y))
        return self

    def predict(self, Xseq, Xt):
        return self.model.predict(Xt)

    def feature_importance(self, columns):
        imp = self.model.booster_.feature_importance(importance_type='gain')
        return dict(zip(columns, imp / max(imp.sum(), 1e-12)))


class CatBoostModel(BaseModel):
    name = 'CatBoost'

    def fit(self, Xseq_tr, Xt_tr, y_tr, Xseq_val=None, Xt_val=None, y_val=None, final=False):
        from catboost import CatBoostRegressor
        params = dict(self.params)
        iters = params.pop('iterations', 500)
        base = dict(random_seed=42, verbose=0, **params)
        if Xt_val is not None and y_val is not None:
            m1 = CatBoostRegressor(iterations=iters, early_stopping_rounds=50, **base)
            m1.fit(Xt_tr, np.ravel(y_tr), eval_set=(Xt_val, np.ravel(y_val)), use_best_model=True)
            best_it = max(10, int(m1.get_best_iteration() or iters) + 1)
            self.fit_info = {'iterations_used': best_it}
            if not final:
                self.model = m1
                return self
            X = np.concatenate([Xt_tr, Xt_val]); y = np.concatenate([y_tr, y_val])
        else:
            best_it, X, y = iters, Xt_tr, y_tr
            self.fit_info = {'iterations_used': best_it}
        self.model = CatBoostRegressor(iterations=best_it, **base).fit(X, np.ravel(y))
        return self

    def predict(self, Xseq, Xt):
        return self.model.predict(Xt)

    def feature_importance(self, columns):
        imp = self.model.get_feature_importance()
        return dict(zip(columns, imp / max(imp.sum(), 1e-12)))

    def save(self, asset_name, task='return_1d'):
        path = os.path.join(MODELS_DIR, f'{artefact_stem(asset_name, self.name, task)}.cbm')
        self.model.save_model(path)
        with open(path + '.json', 'w') as f:
            json.dump({'params': self.params, 'fit_info': self.fit_info}, f)
        return path

    def load(self, asset_name, task='return_1d'):
        from catboost import CatBoostRegressor
        path = os.path.join(MODELS_DIR, f'{artefact_stem(asset_name, self.name, task)}.cbm')
        self.model = CatBoostRegressor(); self.model.load_model(path)
        with open(path + '.json') as f:
            meta = json.load(f)
        self.params, self.fit_info = meta['params'], meta['fit_info']
        return self


# ============================================================ recurrent
class _KerasSequenceModel(BaseModel):
    kind = 'recurrent'
    builder = None            # set in subclass

    def _build(self, seq_len, n_features):
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        return self.builder(seq_len, n_features,
                            units=self.params.get('units', 32),
                            dropout=self.params.get('dropout', 0.2),
                            learning_rate=self.params.get('learning_rate', 1e-3))

    def fit(self, Xseq_tr, Xt_tr, y_tr, Xseq_val=None, Xt_val=None, y_val=None, final=False):
        from tensorflow.keras.callbacks import EarlyStopping
        set_all_seeds(42)
        bs = self.params.get('batch_size', 32)
        epochs = self.params.get('epochs', 80)
        patience = self.params.get('patience', 10)
        seq_len, n_features = Xseq_tr.shape[1], Xseq_tr.shape[2]

        if Xseq_val is not None and y_val is not None:
            m1 = self._build(seq_len, n_features)
            # Only early stopping: phase B has no validation signal, so phase A must not use any other
            # validation-driven schedule (an LR schedule here would make the two fits different procedures).
            cbs = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
            h = m1.fit(Xseq_tr, y_tr, validation_data=(Xseq_val, y_val), epochs=epochs,
                       batch_size=bs, callbacks=cbs, verbose=0, shuffle=False)
            self.history = {k: [float(x) for x in v] for k, v in h.history.items()}
            best_epoch = int(np.argmin(h.history['val_loss'])) + 1
            self.fit_info = {'epochs_used': best_epoch}
            if not final:
                self.model = m1
                return self
            X = np.concatenate([Xseq_tr, Xseq_val]); y = np.concatenate([y_tr, y_val])
        else:
            best_epoch, X, y = epochs, Xseq_tr, y_tr
            self.fit_info = {'epochs_used': best_epoch}
        set_all_seeds(42)
        self.model = self._build(seq_len, n_features)
        self.model.fit(X, y, epochs=best_epoch, batch_size=bs, verbose=0, shuffle=False)
        return self

    def predict(self, Xseq, Xt):
        return np.ravel(self.model.predict(Xseq, verbose=0))

    def save(self, asset_name, task='return_1d'):
        path = os.path.join(MODELS_DIR, f'{artefact_stem(asset_name, self.name, task)}.keras')
        self.model.save(path)
        with open(path + '.json', 'w') as f:
            json.dump({'params': self.params, 'fit_info': self.fit_info, 'history': self.history}, f)
        return path

    def load(self, asset_name, task='return_1d'):
        import logging
        from tensorflow.keras.models import load_model
        logging.getLogger('tensorflow').setLevel(logging.ERROR)     # silence the benign 'retracing' warning at inference
        path = os.path.join(MODELS_DIR, f'{artefact_stem(asset_name, self.name, task)}.keras')
        self.model = load_model(path)
        with open(path + '.json') as f:
            meta = json.load(f)
        self.params, self.fit_info, self.history = meta['params'], meta['fit_info'], meta.get('history')
        return self


class GRUModel(_KerasSequenceModel):
    name = 'GRU'

    @property
    def builder(self):
        from src.models.model_gru import build_gru_model
        return build_gru_model


class LSTMModel(_KerasSequenceModel):
    name = 'LSTM'

    @property
    def builder(self):
        from src.models.model_lstm import build_lstm_model
        return build_lstm_model


# ============================================================ factory
_REGISTRY = {c.name: c for c in (NaiveModel, NaiveMean, ARIMAModel, RidgeModel, RandomForestModel,
                                 LightGBMModel, CatBoostModel, GRUModel, LSTMModel)}


def make_model(name, params=None, **kw):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Known: {list(_REGISTRY)}")
    return _REGISTRY[name](params, **kw) if name == 'Naive' else _REGISTRY[name](params)


def load_trained(name, asset_name, task='return_1d'):
    return make_model(name).load(asset_name, task)


def artefact_exists(name, asset_name, task='return_1d'):
    stem = os.path.join(MODELS_DIR, artefact_stem(asset_name, name, task))
    return any(os.path.exists(stem + ext) for ext in ('.pkl', '.cbm', '.keras'))
