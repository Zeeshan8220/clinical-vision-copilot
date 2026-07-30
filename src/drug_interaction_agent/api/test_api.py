"""Quick in-process test of the Drug Interaction Agent API."""

from fastapi.testclient import TestClient
from main import app

with TestClient(app) as client:
    health = client.get("/health")
    print("Health check:", health.json())

    resp = client.post("/check", json={"medications": ["Warfarin", "Ibuprofen", "Turmeric"]})
    print("Result:", resp.json())
