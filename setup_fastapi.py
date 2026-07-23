import os
files = {}

files['src/radiology_agent/api/main.py'] = '''"""
FastAPI endpoint for the Radiology Agent.

Uses the baseline model (EfficientNet-B0 + Focal Loss + class weighting)
with the tuned decision threshold (0.8) found during Phase 1 experiments,
which gave the best overall balance between NORMAL and PNEUMONIA recall.

Run locally (from repo root):
  uvicorn src.radiology_agent.api.main:app --reload

For quick in-notebook testing without a live server, see test_api.py in
this same folder.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/radiology_agent

import io
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from dataset import EVAL_TRANSFORM, CLASS_NAMES
from model import RadiologyClassifier

CHECKPOINT_PATH = "checkpoints/radiology_epoch5.pt"  # relative to repo root
THRESHOLD = 0.8  # tuned in Phase 1 -- best NORMAL/PNEUMONIA recall balance

app = FastAPI(title="Clinical Vision Copilot — Radiology Agent API")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None


@app.on_event("startup")
def load_model():
    global model
    m = RadiologyClassifier(num_classes=len(CLASS_NAMES))
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    m.load_state_dict(state)
    m.to(device)
    m.eval()
    model = m


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "device": device}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    input_tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pneumonia_prob = float(probs[1])

    prediction = "PNEUMONIA" if pneumonia_prob >= THRESHOLD else "NORMAL"
    confidence = pneumonia_prob if prediction == "PNEUMONIA" else (1 - pneumonia_prob)

    return JSONResponse({
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "pneumonia_probability": round(pneumonia_prob, 4),
        "threshold_used": THRESHOLD,
        "disclaimer": (
            "This is a decision-support / educational tool, not a diagnostic "
            "medical device. Always consult a licensed physician."
        ),
    })
'''

files['src/radiology_agent/api/test_api.py'] = '''"""
Quick test of the FastAPI endpoint WITHOUT running a live server -- uses
FastAPI's built-in TestClient to call the API in-process. This is enough
to verify Phase 1's API works correctly. Live public deployment (Hugging
Face Spaces) happens later in Phase 8.

Run (from repo root):
  python src/radiology_agent/api/test_api.py
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

    # Also test a NORMAL image for comparison
    sample_dir2 = "data/pneumonia/chest_xray/test/NORMAL"
    sample_file2 = os.listdir(sample_dir2)[0]
    sample_path2 = os.path.join(sample_dir2, sample_file2)
    print(f"\nTesting with image: {sample_path2}")

    with open(sample_path2, "rb") as f:
        response2 = client.post("/predict", files={"file": ("xray.jpeg", f, "image/jpeg")})
    print("Prediction response:", response2.json())
'''


for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(content)
    print(f"Written: {path}")

import subprocess
from google.colab import userdata

subprocess.run(["git", "add", "."])
commit = subprocess.run(["git", "commit", "-m", "Add FastAPI endpoint for Radiology Agent (Phase 1 complete)"], capture_output=True, text=True)
print(commit.stdout, commit.stderr)

token = userdata.get('GITHUB_TOKEN')
remote_url = f"https://{token}@github.com/Zeeshan8220/clinical-vision-copilot.git"
push = subprocess.run(["git", "push", remote_url, "main"], capture_output=True, text=True)
print(push.stdout, push.stderr)
