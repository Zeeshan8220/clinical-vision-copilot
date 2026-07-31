"""Quick in-process test of the Differential Diagnosis Agent API."""

from fastapi.testclient import TestClient
from main import app

with TestClient(app) as client:
    health = client.get("/health")
    print("Health check:", health.json())

    resp = client.post("/diagnose", json={
        "symptoms": "fever, cough, chest pain, shortness of breath for 3 days",
        "age": 45, "sex": "M"
    })
    print("Result:", resp.json())
