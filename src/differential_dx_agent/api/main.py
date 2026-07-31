"""FastAPI endpoint for the Differential Diagnosis Agent."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from differential_dx import get_differential_diagnosis

app = FastAPI(title="Clinical Vision Copilot — Differential Diagnosis Agent API")


class PatientSymptoms(BaseModel):
    symptoms: str
    age: Optional[int] = None
    sex: Optional[str] = None
    history: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/diagnose")
def diagnose(payload: PatientSymptoms):
    result = get_differential_diagnosis(
        symptoms=payload.symptoms, age=payload.age, sex=payload.sex, history=payload.history
    )
    result["disclaimer"] = (
        "This is an AI-generated differential diagnosis list for decision-support "
        "and educational purposes only. It is NOT a confirmed diagnosis and is NOT "
        "a substitute for evaluation by a licensed physician."
    )
    return result
