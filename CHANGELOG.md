# Changelog

All notable changes to the Crypto-Forex Prediction System will be documented in this file.

## [Week 1] — Stop the Bleeding (Correctness Fixes)

### Critical Fixes
- **Fixed data leakage in 3 experiment scripts** — `experiment_dropout.py`, `experiment_seq_length.py`, and `experiment_lstm_units.py` all used `validation_data=(X_test, y_test)`, contaminating hyperparameter selection with held-out test data. Now use proper `(X_val, y_val)` for monitoring, test only for final evaluation. *Audit finding: Part 4, Training Pipeline.*
- **Fixed default model path in `prediction.py`** — Changed from `btc_lstm_best.keras` (trained with leaky validation) to `btc_lstm_final.keras` (trained with correct 2-phase approach). *Audit finding: Part 4.*
- **Disabled Silver LSTM** — Silver LSTM (R²=-0.849, 27.5% directional accuracy — below coin-flip) is now marked as disabled in `MODEL_STATUS` config. Falls back to Linear Regression. *Audit finding: Part 3, Model Architecture.*

### High-Priority Fixes
- **Pinned all dependencies** — `requirements.txt` now uses exact versions instead of `>=` minimums. Added missing: `statsmodels`, `joblib`, `lightgbm`, `python-dotenv`. Safety snapshot saved in `requirements-lock.txt`.
- **Added reproducibility seeds** — New `src/utils/reproducibility.py` provides `set_all_seeds(42)` covering Python, NumPy, TF, and PYTHONHASHSEED. Added to all experiment scripts.
- **Fixed `seq_len` inconsistency** — Config specified `seq_len=30` but pipeline defaulted to 60. Now all paths use `BEST_LSTM_CONFIG['seq_len']` dynamically.

### Medium-Priority Fixes
- **Fixed broken test import** — `test_preprocessing.py` imported from `src.preprocessing` instead of `src.data.preprocessing`.
- **Removed hardcoded `n_features=16`** — All 5 occurrences across `evaluation.py`, `walk_forward.py`, `ensemble_model.py`, `train_final_btc.py`, and `streamlit_app.py` now use dynamic `scaler.n_features_in_`.
- **Added deprecation warning to `training.py`** — Legacy script has known data leakage; users are directed to `train_final_btc.py`.

### Low-Priority Fixes
- **Replaced empty `test_functions.py`** — Now contains real tests for RSI range, SMA correctness, Bollinger Band ordering, MACD structure, feature count, and no-lookahead verification.
- **Removed unused `import requests`** from `data_collection.py`.

### Infrastructure
- **Fixed `.gitignore`** — Added `.env` file entry (was only ignoring `.env/` directory, leaving actual `.env` files exposed).
- **Created `.env.example`** — Template for future FRED/NewsAPI keys.
- **Created `requirements-lock.txt`** — Full `pip freeze` safety snapshot before changes.
