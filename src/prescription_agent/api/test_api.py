"""Quick in-process test of the Prescription Writer Agent API."""

from fastapi.testclient import TestClient
from main import app

with TestClient(app) as client:
    health = client.get("/health")
    print("Health check:", health.json())

    resp = client.post("/prescribe", json={
        "diagnosis": "Community-acquired pneumonia (mild, outpatient)",
        "age": 45, "sex": "M", "allergies": "None known", "current_medications": "None"
    })
    print("Result:", resp.json())
