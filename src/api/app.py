import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.inference.prediction import predict_next_day
from config import ASSET_CONFIG, MODEL_STATUS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto & Forex Prediction API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    asset: str
    predicted_price: float
    best_model: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/models")
def list_models():
    return {"models": MODEL_STATUS}

@app.get("/predict/{asset}", response_model=PredictionResponse)
def predict_asset(asset: str):
    valid_assets = [a for a in ASSET_CONFIG.keys()]
    # Simple mapping case-insensitive
    asset_map = {a.lower(): a for a in valid_assets}
    
    if asset.lower() not in asset_map:
        raise HTTPException(status_code=404, detail=f"Asset {asset} not supported. Supported assets: {valid_assets}")
        
    actual_asset_name = asset_map[asset.lower()]
    
    try:
        logger.info(f"Received prediction request for {actual_asset_name}")
        # Run prediction
        price = predict_next_day(actual_asset_name)
        if price is None:
            raise ValueError("Prediction returned None")
            
        best_model = MODEL_STATUS.get(actual_asset_name, {}).get("best_model", "Unknown")
        
        return PredictionResponse(
            asset=actual_asset_name,
            predicted_price=float(price),
            best_model=best_model
        )
    except Exception as e:
        logger.error(f"Error predicting {actual_asset_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
