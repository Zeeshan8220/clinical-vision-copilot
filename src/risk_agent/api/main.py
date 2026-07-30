"""
FastAPI endpoint for the Risk Score Agent.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

from dataset import CATEGORICAL_COLS

MODEL_PATH = "risk_model.joblib"

app = FastAPI(title="Clinical Vision Copilot — Risk Score Agent API")

model = None
explainer = None
train_columns = None


class PatientData(BaseModel):
    Age: int
    Sex: Literal["M", "F"]
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"]
    RestingBP: int
    Cholesterol: int
    FastingBS: Literal[0, 1]
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: int
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float
    ST_Slope: Literal["Up", "Flat", "Down"]


@app.on_event("startup")
def load_model():
    global model, explainer, train_columns
    model = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    train_columns = model.get_booster().feature_names


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(patient: PatientData):
    row = pd.DataFrame([patient.dict()])
    row_encoded = pd.get_dummies(row, columns=CATEGORICAL_COLS)
    row_aligned = row_encoded.reindex(columns=train_columns, fill_value=0)

    prob = float(model.predict_proba(row_aligned)[0, 1])
    risk_level = "High" if prob >= 0.6 else ("Medium" if prob >= 0.3 else "Low")

    shap_values = explainer(row_aligned)
    contributions = list(zip(train_columns, shap_values.values[0]))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_factors = [
        {"feature": name, "impact": round(float(val), 3)}
        for name, val in contributions[:5]
    ]

    return {
        "risk_probability": round(prob, 4),
        "risk_level": risk_level,
        "top_factors": top_factors,
        "disclaimer": "This is a decision-support / educational tool, not a diagnostic medical device. Always consult a licensed physician.",
    }
