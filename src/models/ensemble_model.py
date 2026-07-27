"""
Stacked Ensemble using Expanding Window CV (Walk-Forward Validation).

Week 6: Upgrade from a hardcoded weighted average to a machine learning Meta-Model.
Uses Walk-Forward Cross Validation to generate out-of-fold (OOF) predictions
to train the meta-model (Ridge Regression) without data leakage.

Combines:
- GRU (Sequence modeling)
- LightGBM (Tabular feature interactions)
- Ridge (Linear baseline)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import PROCESSED_DATA_DIR, MODELS_DIR, BEST_GRU_CONFIG, BEST_LGBM_CONFIG, CV_FOLDS
from src.models.model_gru import build_gru_model
from src.models.model_lgbm import build_lgbm_model
from src.evaluation.cross_validation import generate_oof_predictions
from src.utils.inverse_transform import reconstruct_price
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_prefix(asset):
    return 'btc' if asset == 'Bitcoin' else asset.lower()

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE_USD': rmse, 'MAE_USD': mae, 'R2': r2}

def load_data(asset):
    """Load train, val, and test sequences and concatenate train+val for OOF generation."""
    prefix = get_prefix(asset)
    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_train.npy"))
    X_val   = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_val.npy"))
    y_val   = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_val.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_test.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_test.npy"))
    
    # Merge train and val to use for cross-validation
    X_cv = np.concatenate([X_train, X_val], axis=0)
    y_cv = np.concatenate([y_train, y_val], axis=0)
    
    # Get test dates for plotting
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_train_scaled.csv'), index_col=0, parse_dates=True)
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_val_scaled.csv'), index_col=0, parse_dates=True)
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_test_scaled.csv'), index_col=0, parse_dates=True)
    
    full_df = pd.concat([train_df, val_df, test_df])
    seq_len = BEST_GRU_CONFIG['seq_len']
    L_train = len(train_df)
    L_val = len(val_df)
    # Get test dates for plotting (matches the length of y_test)
    test_dates = test_df.index[-len(y_test):]
    
    # Get scaler and target_idx
    scaler = joblib.load(os.path.join(MODELS_DIR, f'{prefix}_scaler.pkl'))
    target_idx = list(full_df.columns).index('log_return') if 'log_return' in full_df.columns else list(full_df.columns).index('price')
    
    return X_cv, y_cv, X_test, y_test, test_dates, scaler, target_idx

def run_stacked_ensemble(asset):
    """Run Expanding Window CV and Stacking for one asset."""
    print(f"\n{'='*70}")
    print(f"  {asset.upper()} — Stacked Ensemble via Walk-Forward CV")
    print(f"{'='*70}")
    
    X_cv, y_cv, X_test, y_test, test_dates, scaler, target_idx = load_data(asset)
    n_features = X_cv.shape[2]
    
    # --- GRU OOF Configuration ---
    def gru_builder():
        return build_gru_model(seq_len=BEST_GRU_CONFIG['seq_len'], n_features=n_features,
                               learning_rate=BEST_GRU_CONFIG['learning_rate'],
                               gru_units=BEST_GRU_CONFIG['gru_units'],
                               dense_units=BEST_GRU_CONFIG['dense_units'],
                               dropout_rate=BEST_GRU_CONFIG['dropout_rate'])
                               
    def gru_trainer(model, X_t, y_t, X_v, y_v):
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
        model.fit(X_t, y_t, validation_data=(X_v, y_v), epochs=30, batch_size=BEST_GRU_CONFIG['batch_size'], callbacks=[es], verbose=0)
        return model
        
    def gru_predictor(model, X_pred):
        return model.predict(X_pred, verbose=0)
        
    # --- LightGBM OOF Configuration ---
    def lgbm_builder():
        return build_lgbm_model(n_estimators=BEST_LGBM_CONFIG['n_estimators'],
                                learning_rate=BEST_LGBM_CONFIG['learning_rate'],
                                max_depth=BEST_LGBM_CONFIG['max_depth'],
                                num_leaves=BEST_LGBM_CONFIG['num_leaves'],
                                subsample=BEST_LGBM_CONFIG['subsample'],
                                colsample_bytree=BEST_LGBM_CONFIG['colsample_bytree'])
                                
    def lgbm_trainer(model, X_t, y_t, X_v, y_v):
        X_t_flat = X_t.reshape(X_t.shape[0], -1)
        X_v_flat = X_v.reshape(X_v.shape[0], -1)
        model.fit(X_t_flat, y_t.ravel(), eval_set=[(X_v_flat, y_v.ravel())], callbacks=[])
        return model
        
    def lgbm_predictor(model, X_pred):
        X_pred_flat = X_pred.reshape(X_pred.shape[0], -1)
        return model.predict(X_pred_flat)
        
    # --- Ridge Baseline OOF Configuration ---
    def ridge_builder():
        return Ridge(alpha=1.0)
        
    def ridge_trainer(model, X_t, y_t, X_v, y_v):
        X_t_flat = X_t.reshape(X_t.shape[0], -1)
        model.fit(X_t_flat, y_t.ravel())
        return model
        
    def ridge_predictor(model, X_pred):
        X_pred_flat = X_pred.reshape(X_pred.shape[0], -1)
        return model.predict(X_pred_flat)

    # 1. Generate OOF Predictions
    print(f"  Generating Out-Of-Fold (OOF) predictions ({CV_FOLDS} folds)...")
    
    print("    -> GRU (Deep Learning)")
    gru_oof_preds, oof_true = generate_oof_predictions(gru_builder, gru_trainer, gru_predictor, X_cv, y_cv, n_splits=CV_FOLDS)
    
    print("    -> LightGBM (Tree Ensemble)")
    lgbm_oof_preds, _ = generate_oof_predictions(lgbm_builder, lgbm_trainer, lgbm_predictor, X_cv, y_cv, n_splits=CV_FOLDS)
    
    print("    -> Ridge (Linear Baseline)")
    ridge_oof_preds, _ = generate_oof_predictions(ridge_builder, ridge_trainer, ridge_predictor, X_cv, y_cv, n_splits=CV_FOLDS)
    
    # 2. Train Meta-Model (Stacker)
    print("\n  Training Meta-Model (Ridge Regression) on OOF predictions...")
    # Stack OOF predictions as features (Samples x Models)
    OOF_X = np.column_stack((gru_oof_preds, lgbm_oof_preds, ridge_oof_preds))
    meta_model = Ridge(alpha=1.0, positive=True) # positive=True forces non-negative weights (like an ensemble)
    meta_model.fit(OOF_X, oof_true)
    
    weights = meta_model.coef_
    total_weight = np.sum(weights) + 1e-8
    print(f"    Learned Weights — GRU: {weights[0]/total_weight:.1%}, LGBM: {weights[1]/total_weight:.1%}, Ridge: {weights[2]/total_weight:.1%}")
    
    # 3. Train final base models on FULL CV data
    print("\n  Training final base models on all CV data...")
    final_gru = gru_builder()
    final_gru.fit(X_cv, y_cv, epochs=15, batch_size=BEST_GRU_CONFIG['batch_size'], verbose=0)
    
    final_lgbm = lgbm_builder()
    final_lgbm.fit(X_cv.reshape(X_cv.shape[0], -1), y_cv.ravel())
    
    final_ridge = ridge_builder()
    final_ridge.fit(X_cv.reshape(X_cv.shape[0], -1), y_cv.ravel())
    
    # 4. Predict on Blind Test Set
    print("  Evaluating on Blind Test Set...")
    pred_gru = final_gru.predict(X_test, verbose=0).ravel()
    pred_lgbm = final_lgbm.predict(X_test.reshape(X_test.shape[0], -1)).ravel()
    pred_ridge = final_ridge.predict(X_test.reshape(X_test.shape[0], -1)).ravel()
    
    TEST_X = np.column_stack((pred_gru, pred_lgbm, pred_ridge))
    pred_meta = meta_model.predict(TEST_X)
    
    # 5. Inverse Transform for actual USD evaluation
    y_test_real = reconstruct_price(y_test, X_test, scaler, target_idx)
    pred_gru_real = reconstruct_price(pred_gru.reshape(-1,1), X_test, scaler, target_idx)
    pred_lgbm_real = reconstruct_price(pred_lgbm.reshape(-1,1), X_test, scaler, target_idx)
    pred_meta_real = reconstruct_price(pred_meta.reshape(-1,1), X_test, scaler, target_idx)
    
    m_gru = compute_metrics(y_test_real, pred_gru_real)
    m_lgbm = compute_metrics(y_test_real, pred_lgbm_real)
    m_meta = compute_metrics(y_test_real, pred_meta_real)
    
    print(f"\n  {'Model':<25} {'RMSE (USD)':>12} {'MAE (USD)':>12} {'R²':>10}")
    print(f"  {'-'*62}")
    for name, m in [('GRU (Standalone)', m_gru), ('LightGBM (Standalone)', m_lgbm), ('Stacked Meta-Model', m_meta)]:
        print(f"  {name:<25} ${m['RMSE_USD']:>10,.2f} ${m['MAE_USD']:>10,.2f} {m['R2']:>9.4f}")
        
    best_standalone_rmse = min(m_gru['RMSE_USD'], m_lgbm['RMSE_USD'])
    if m_meta['RMSE_USD'] < best_standalone_rmse:
        improvement = (1 - m_meta['RMSE_USD'] / best_standalone_rmse) * 100
        print(f"\n  ✅ Meta-Model WINS — {improvement:.2f}% improvement over best standalone")
        winner = 'Meta-Model'
    else:
        print(f"\n  ❌ Meta-Model LOSES — Standalone model performed better on this specific test set")
        winner = 'GRU' if m_gru['RMSE_USD'] < m_lgbm['RMSE_USD'] else 'LightGBM'
        
    # 6. Plot
    prefix = get_prefix(asset)
    fig, axes = plt.subplots(1, 1, figsize=(14, 6))
    axes.plot(test_dates, y_test_real, label='Actual Price', color='#F7931A', linewidth=2.5)
    axes.plot(test_dates, pred_gru_real, label='GRU (Standalone)', color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    axes.plot(test_dates, pred_lgbm_real, label='LightGBM (Standalone)', color='#2196F3', linestyle='--', linewidth=1.5)
    axes.plot(test_dates, pred_meta_real, label=f'Stacked Meta-Model', color='#E91E63', linewidth=2)
    axes.set_title(f'{asset} — Stacked Ensemble vs Standalone Models', fontsize=14, fontweight='bold')
    axes.set_ylabel('Price (USD)')
    axes.legend(fontsize=10)
    axes.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(RESULTS_DIR, f'{prefix}_stacked_ensemble.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  ✅ Plot saved to {plot_path}")
    
    # Save the meta model
    joblib.dump(meta_model, os.path.join(MODELS_DIR, f'{prefix}_meta_model.pkl'))
    
    return {
        'Asset': asset,
        'GRU_RMSE': m_gru['RMSE_USD'],
        'LGBM_RMSE': m_lgbm['RMSE_USD'],
        'Meta_RMSE': m_meta['RMSE_USD'],
        'Meta_R2': m_meta['R2'],
        'Winner': winner
    }

if __name__ == "__main__":
    all_results = []
    for asset in ['Bitcoin', 'Gold', 'Silver']:
        res = run_stacked_ensemble(asset)
        all_results.append(res)
        
    print(f"\n\n{'='*80}")
    print("  FINAL META-MODEL COMPARISON — All Assets")
    print(f"{'='*80}")
    df = pd.DataFrame(all_results)
    print(f"\n  {'Asset':<10} {'Best Base RMSE':>16} {'Meta-Model RMSE':>18} {'Winner':>15}")
    print(f"  {'-'*65}")
    for _, row in df.iterrows():
        best_base = min(row['GRU_RMSE'], row['LGBM_RMSE'])
        print(f"  {row['Asset']:<10} ${best_base:>15,.2f} ${row['Meta_RMSE']:>17,.2f} {row['Winner']:>15}")
