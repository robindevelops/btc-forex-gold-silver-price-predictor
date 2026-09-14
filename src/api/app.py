"""
FastAPI service exposing the served forecast per asset.

    uvicorn src.api.app:app --reload
    GET /health
    GET /models                       → served forecast (Combined), its members, the CV-selected single model and
                                        held-out metrics per asset
    GET /predict/{asset}              → next-day prediction, asset = bitcoin | gold | silver (default = Combined
                                        forecast of every trained model, with each member's own prediction)
    GET /predict/{asset}?model=GRU    → prediction with one specific trained model
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
from src.inference.prediction import predict_next_day, available_models, combined_members, COMBINED

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Asset Next-Day Return Prediction API", version="2.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class MemberPrediction(BaseModel):
    model: str
    predicted_price: float
    predicted_return_pct: float
    direction: str


class PredictionResponse(BaseModel):
    asset: str
    as_of_date: str
    target_date: Optional[str] = None
    horizon_days: int
    current_price: float
    predicted_price: float
    predicted_return_pct: float
    direction: str
    model_used: str
    is_served_model: bool
    cv_selected_model: Optional[str] = None
    individual: Optional[List[MemberPrediction]] = None
    uncertainty_band: Optional[List[float]] = None
    data_source: Optional[str] = None
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
    return {a: {**status.get(a, {}), 'served_forecast': COMBINED, 'combined_members': combined_members(a),
                'available_models': available_models(a)} for a in ASSET_CONFIG}


@app.get("/predict/{asset}", response_model=PredictionResponse)
def predict_asset(asset: str, model: Optional[str] = Query(None, description="One trained model name; default = the Combined forecast")):
    name = _resolve(asset)
    if model and model not in available_models(name) + [COMBINED]:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not available for {name}. Available: {available_models(name) + [COMBINED]}")
    try:
        return PredictionResponse(**predict_next_day(name, model or COMBINED))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model or data artefacts missing: {e}")
    except Exception as e:
        logger.exception(f"Error predicting {name}")
        raise HTTPException(status_code=500, detail=str(e))

