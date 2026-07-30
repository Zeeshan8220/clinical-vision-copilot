"""
FastAPI endpoint for the Drug Interaction Agent.
Hybrid: verified curated database first, LLM (Groq) secondary opinion
for pairs not in the database (clearly labeled as unverified).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from interaction_checker import check_interactions_hybrid

app = FastAPI(title="Clinical Vision Copilot — Drug Interaction Agent API")


class MedicationList(BaseModel):
    medications: List[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/check")
def check(payload: MedicationList):
    results = check_interactions_hybrid(payload.medications, use_llm=True)
    return {
        "medications_checked": payload.medications,
        "interactions_found": results,
        "disclaimer": (
            "This is a decision-support / educational tool, not a diagnostic "
            "medical device. \'verified_database\' results are from a small "
            "curated reference list; \'ai_generated_unverified\' results are "
            "AI-generated and NOT clinically verified. Always consult a "
            "licensed pharmacist or physician."
        ),
    }
