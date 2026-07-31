"""Quick in-process test of the Knowledge/RAG Agent API."""

from fastapi.testclient import TestClient
from main import app

with TestClient(app) as client:
    health = client.get("/health")
    print("Health check:", health.json())

    resp = client.post("/ask", json={"question": "What is first-line treatment for type 2 diabetes?"})
    print("Result:", resp.json())
