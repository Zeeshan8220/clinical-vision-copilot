"""
Quick in-process test of the Risk Score Agent API.
"""

from fastapi.testclient import TestClient
from main import app

with TestClient(app) as client:
    health = client.get("/health")
    print("Health check:", health.json())

    high_risk_patient = {
        "Age": 58, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140,
        "Cholesterol": 289, "FastingBS": 1, "RestingECG": "ST", "MaxHR": 120,
        "ExerciseAngina": "Y", "Oldpeak": 2.0, "ST_Slope": "Flat"
    }
    resp1 = client.post("/predict", json=high_risk_patient)
    print("\nHigh-risk-looking patient:")
    print(resp1.json())

    low_risk_patient = {
        "Age": 35, "Sex": "F", "ChestPainType": "ATA", "RestingBP": 110,
        "Cholesterol": 180, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 175,
        "ExerciseAngina": "N", "Oldpeak": 0.0, "ST_Slope": "Up"
    }
    resp2 = client.post("/predict", json=low_risk_patient)
    print("\nLow-risk-looking patient:")
    print(resp2.json())
