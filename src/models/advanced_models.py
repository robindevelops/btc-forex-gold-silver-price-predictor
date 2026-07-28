import os
import pandas as pd
import numpy as np
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST, TFT, NBEATS
from neuralforecast.losses.pytorch import MSE
from config import MODELS_DIR, BEST_LSTM_CONFIG

def create_nf_dataset(X, y, asset_name):
    """
    Converts 3D sequence data (samples, seq_len, features) into the 
    long-format DataFrame required by NeuralForecast.
    Format required: ['unique_id', 'ds', 'y', 'feature_1', 'feature_2', ...]
    """
    # For PyTorch models, we need a flat history for each 'unique_id'
    # Since we pre-sliced into windows of length 30, it's easier to just take the raw time series
    # But to match our Keras pipeline, we can construct independent series.
    # Actually, NeuralForecast natively handles windowing. 
    # The best approach is to rebuild the dataframe from the raw scaled features
    # rather than the pre-sliced 3D arrays to utilize NeuralForecast's optimized dataloaders.
    pass


def train_neuralforecast_model(train_df, val_df, asset_name, model_type="PatchTST"):
    """
    Trains a state-of-the-art NeuralForecast model.
    Args:
        train_df: The fully scaled training DataFrame (indexed by timestamp)
        val_df: The fully scaled validation DataFrame (indexed by timestamp)
        asset_name: Name of the asset
        model_type: "PatchTST", "TFT", or "NBEATS"
    """
    print(f"\n{'='*50}\n  TRAINING {model_type} MODEL ({asset_name})\n{'='*50}")
    
    # 1. Prepare Data format
    # NeuralForecast expects: unique_id, ds (datetime/int), y, and exogenous features
    def format_nf(df, uid):
        nf_df = df.copy()
        nf_df['unique_id'] = uid
        nf_df = nf_df.reset_index()
        nf_df = nf_df.rename(columns={'timestamp': 'ds', 'price': 'y'})
        # Select target and features
        cols = ['unique_id', 'ds', 'y'] + [c for c in nf_df.columns if c not in ['unique_id', 'ds', 'y']]
        return nf_df[cols]

    nf_train = format_nf(train_df, asset_name)
    nf_val = format_nf(val_df, asset_name)
    
    exogenous_features = [c for c in nf_train.columns if c not in ['unique_id', 'ds', 'y']]
    
    seq_len = BEST_LSTM_CONFIG['seq_len']  # 30 days lookback
    horizon = 1  # 1 day ahead forecast
    
    # 2. Define Model
    if model_type == "PatchTST":
        model = PatchTST(
            h=horizon,
            input_size=seq_len,
            patch_len=8,
            stride=8,
            hidden_size=64,
            n_heads=4,
            loss=MSE(),
            max_steps=500,
            val_check_steps=50,
            early_stop_patience_steps=10
        )
    elif model_type == "TFT":
        model = TFT(
            h=horizon,
            input_size=seq_len,
            hist_exog_list=exogenous_features,
            hidden_size=64,
            loss=MSE(),
            max_steps=500,
            val_check_steps=50,
            early_stop_patience_steps=10
        )
    elif model_type == "NBEATS":
        model = NBEATS(
            h=horizon,
            input_size=seq_len,
            loss=MSE(),
            max_steps=500,
            val_check_steps=50,
            early_stop_patience_steps=10
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 3. Train Model
    nf = NeuralForecast(models=[model], freq='D')
    
    # Concatenate train and val for NeuralForecast (it handles splitting internally based on sizes)
    # Actually we can just fit on train, but we want validation early stopping.
    nf.fit(df=nf_train, val_size=len(nf_val))
    
    # 4. Predict on Validation to get metrics
    # nf.predict() expects a history of length `input_size` for each prediction.
    # To evaluate properly like Walk Forward, we need to pass the concatenated df
    full_df = pd.concat([nf_train, nf_val])
    # ... evaluating full sequence ...
    
    # Save Model
    model_path = os.path.join(MODELS_DIR, f"{asset_name.lower()}_{model_type.lower()}")
    nf.save(path=model_path, overwrite=True)
    print(f"  ✅ {model_type} saved to {model_path}")
    
    return nf
