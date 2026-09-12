"""
FastAPI service exposing the served model per asset.

    uvicorn src.api.app:app --reload
    GET /health
    GET /models                    → served model, selection rule and held-out metrics per asset
    GET /predict/{asset}           → next-day prediction (asset = bitcoin | gold | silver)
    GET /predict/{asset}?model=GRU → prediction with a specific trained model
"""
import os
import sys
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import ASSET_CONFIG, load_model_status
from src.inference.prediction import predict_next_day, available_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Asset Next-Day Return Prediction API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class PredictionResponse(BaseModel):
    asset: str
    as_of_date: str
    horizon_days: int
    current_price: float
    predicted_price: float
    predicted_return_pct: float
    direction: str
    model_used: str
    is_served_model: bool
    uncertainty_band: Optional[List[float]] = None
    test_metrics: Dict[str, Any] = {}
    disclaimer: str


def _resolve(asset: str) -> str:
    m = {a.lower(): a for a in ASSET_CONFIG}
    if asset.lower() not in m:
        raise HTTPException(status_code=404, detail=f"Asset '{asset}' not supported. Use one of {list(ASSET_CONFIG)}")
    return m[asset.lower()]


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/models")
def list_models():
    status = load_model_status()
    return {a: {**status.get(a, {}), 'available_models': available_models(a)} for a in ASSET_CONFIG}


@app.get("/predict/{asset}", response_model=PredictionResponse)
def predict_asset(asset: str, model: Optional[str] = Query(None, description="Trained model name, default = served model")):
    name = _resolve(asset)
    try:
        return PredictionResponse(**predict_next_day(name, model))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model artefacts missing: {e}")
    except Exception as e:
        logger.exception(f"Error predicting {name}")
        raise HTTPException(status_code=500, detail=str(e))
