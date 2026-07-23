"""
FastAPI endpoint for the Radiology Agent.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from dataset import EVAL_TRANSFORM, CLASS_NAMES
from radiology_model import RadiologyClassifier

CHECKPOINT_PATH = "checkpoints/radiology_epoch5.pt"
THRESHOLD = 0.8

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
        "disclaimer": "This is a decision-support / educational tool, not a diagnostic medical device. Always consult a licensed physician.",
    })
