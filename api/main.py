"""
FastAPI backend for the crop yield predictor.

Run locally:
    uvicorn api.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Crop Yield Predictor API",
    description="Predicts crop yield (tonnes/hectare) from soil, weather, "
                 "and nutrient inputs. Uses a separate model per crop.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/yield_model.joblib"
bundle = joblib.load(MODEL_PATH)
per_crop_models = bundle["models"]  # {crop_name: {"pipeline":..., "model_name":..., "metrics":...}}


class PredictionRequest(BaseModel):
    crop: str = Field(..., examples=["Rice"])
    region: str = Field(..., examples=["North"])
    soil_type: str = Field(..., examples=["Alluvial"])
    rainfall_mm: float = Field(..., ge=0, le=3000)
    temperature_c: float = Field(..., ge=-10, le=55)
    humidity_pct: float = Field(..., ge=0, le=100)
    nitrogen_kg_ha: float = Field(..., ge=0, le=300)
    phosphorus_kg_ha: float = Field(..., ge=0, le=300)
    potassium_kg_ha: float = Field(..., ge=0, le=300)
    ph: float = Field(..., ge=0, le=14)
    area_hectares: float = Field(..., gt=0, le=1000)


class PredictionResponse(BaseModel):
    predicted_yield_tonnes_per_hectare: float
    predicted_total_yield_tonnes: float
    model_used: str
    crop: str


@app.get("/")
def root():
    return {
        "message": "Crop Yield Predictor API is running",
        "crops_available": list(per_crop_models.keys()),
        "docs": "/docs",
    }


@app.get("/metadata")
def metadata():
    """Valid categorical values, for building a frontend dropdown."""
    return {
        "crops": bundle["crops"],
        "regions": bundle["regions"],
        "soil_types": bundle["soil_types"],
    }


@app.get("/metrics/{crop}")
def crop_metrics(crop: str):
    """Model performance for a specific crop's model."""
    if crop not in per_crop_models:
        raise HTTPException(status_code=404, detail=f"No model for crop '{crop}'")
    entry = per_crop_models[crop]
    return {"crop": crop, "model_name": entry["model_name"], "metrics": entry["metrics"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if req.crop not in per_crop_models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown crop '{req.crop}'. Valid options: {list(per_crop_models.keys())}",
        )

    entry = per_crop_models[req.crop]
    pipeline = entry["pipeline"]

    # crop is not a feature for the per-crop model — it's dropped here
    input_dict = req.model_dump()
    input_dict.pop("crop")
    input_df = pd.DataFrame([input_dict])

    pred_per_ha = float(pipeline.predict(input_df)[0])

    return PredictionResponse(
        predicted_yield_tonnes_per_hectare=round(pred_per_ha, 2),
        predicted_total_yield_tonnes=round(pred_per_ha * req.area_hectares, 2),
        model_used=entry["model_name"],
        crop=req.crop,
    )