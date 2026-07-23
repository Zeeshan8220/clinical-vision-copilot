"""
Quick test of the FastAPI endpoint using TestClient (no live server needed).
"""

import os
from fastapi.testclient import TestClient
from main import app

with TestClient(app) as client:
    health = client.get("/health")
    print("Health check:", health.json())

    sample_dir = "data/pneumonia/chest_xray/test/PNEUMONIA"
    sample_file = os.listdir(sample_dir)[0]
    sample_path = os.path.join(sample_dir, sample_file)
    print(f"Testing with image: {sample_path}")

    with open(sample_path, "rb") as f:
        response = client.post("/predict", files={"file": ("xray.jpeg", f, "image/jpeg")})
    print("Prediction response:", response.json())

    sample_dir2 = "data/pneumonia/chest_xray/test/NORMAL"
    sample_file2 = os.listdir(sample_dir2)[0]
    sample_path2 = os.path.join(sample_dir2, sample_file2)
    print(f"\nTesting with image: {sample_path2}")

    with open(sample_path2, "rb") as f:
        response2 = client.post("/predict", files={"file": ("xray.jpeg", f, "image/jpeg")})
    print("Prediction response:", response2.json())
