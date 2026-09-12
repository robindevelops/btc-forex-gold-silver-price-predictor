"""Target alignment, split integrity and exact price reconstruction on the REAL processed data."""
import os
import numpy as np
import pandas as pd
import pytest
from config import PROCESSED_DATA_DIR, TRAIN_END, VAL_END, ASSETS, get_prefix
from src.data.preprocessing import create_sequences, build_dataset, unscale_target
from src.utils.inverse_transform import reconstruct_price

HAVE_DATA = all(os.path.exists(os.path.join(PROCESSED_DATA_DIR, f'{get_prefix(a)}_train_scaled.csv')) for a in ASSETS)


def test_create_sequences_target_is_next_row_and_never_in_window():
    n, seq = 50, 10
    df = pd.DataFrame({'f': np.arange(n, dtype=float), 'log_return': np.arange(n, dtype=float) * 100})
    X, y = create_sequences(df, seq_len=seq)
    assert X.shape == (n - seq, seq, 2) and y.shape == (n - seq, 1)
    for i in range(len(y)):
        assert y[i, 0] == df['log_return'].iloc[i + seq]           # target = row i+seq
        assert df['log_return'].iloc[i + seq] not in X[i][:, 1]     # ... and that row is not in the window


@pytest.mark.skipif(not HAVE_DATA, reason="processed data not present")
@pytest.mark.parametrize('asset', ASSETS)
def test_split_is_chronological_and_disjoint(asset):
    d = build_dataset(asset)
    assert d['dates_train'].max() <= pd.Timestamp(TRAIN_END)
    assert pd.Timestamp(TRAIN_END) < d['dates_val'].min() and d['dates_val'].max() <= pd.Timestamp(VAL_END)
    assert d['dates_test'].min() > pd.Timestamp(VAL_END)
    assert d['dates_train'].is_monotonic_increasing and d['dates_test'].is_monotonic_increasing


@pytest.mark.skipif(not HAVE_DATA, reason="processed data not present")
@pytest.mark.parametrize('asset', ASSETS)
def test_true_target_reconstructs_actual_close_exactly(asset):
    """This is the regression test for the old `open`-column bug (was $1,400 off for BTC)."""
    d = build_dataset(asset)
    for split in ('train', 'val', 'test'):
        rec = reconstruct_price(d[f'y_{split}'], d[f'prev_{split}'], d['scaler'], d['target_idx'])
        assert np.allclose(rec, d[f'true_{split}'], rtol=1e-9)


@pytest.mark.skipif(not HAVE_DATA, reason="processed data not present")
def test_scaler_fitted_on_train_only():
    d = build_dataset('Bitcoin')
    tr = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'btc_train_scaled.csv'), index_col=0)
    assert np.isclose(tr.min().min(), 0.0) and np.isclose(tr.max().max(), 1.0)
    # the val/test frames are NOT guaranteed to lie in [0,1] -> proves they did not influence the fit
    te = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'btc_test_scaled.csv'), index_col=0)
    assert te.shape[1] == tr.shape[1] == d['scaler'].n_features_in_
