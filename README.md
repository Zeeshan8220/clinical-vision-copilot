# Clinical Vision Copilot

Multi-agent AI decision-support system for medical case analysis. Built solo,
zero-budget, using free tools (Google Colab, Kaggle datasets, Hugging Face)
and public datasets only.

**This is a decision-support / educational tool. It is not a diagnostic
medical device and is not a substitute for a licensed doctor's judgement.**

## Architecture

Six agents, orchestrated with LangGraph:

| # | Agent | Status | Folder |
|---|-------|--------|--------|
| 1 | Radiology Agent | ✅ Complete — model, Grad-CAM, FastAPI endpoint (Phase 1) | `src/radiology_agent/` |
| 2 | Risk Score Agent | ⏳ Not started (Phase 2) | `src/risk_agent/` |
| 3 | Drug Interaction Agent | ⏳ Port from MedGenius (Phase 3) | `src/drug_interaction_agent/` |
| 4 | Differential Diagnosis Agent | ⏳ Port from MedGenius (Phase 4) | `src/differential_dx_agent/` |
| 5 | Prescription Writer Agent | ⏳ Port from MedGenius (Phase 5) | `src/prescription_agent/` |
| 6 | Knowledge/RAG Agent | ⏳ Not started (Phase 6) | `src/knowledge_agent/` |
| — | Orchestrator | ⏳ Not started (Phase 7) | `src/orchestrator/` |

## Phase 1 — Radiology Agent (Chest X-ray Pneumonia Detection)

**Task:** Binary classification (NORMAL vs PNEUMONIA) on chest X-rays.

**Two model variants were built and compared:**

| Model | Backbone | Best Test AUC | Best Accuracy | Notes |
|---|---|---|---|---|
| Baseline | EfficientNet-B0 (ImageNet-pretrained) + Focal Loss | 0.96 | 0.91 (@ threshold 0.8) | Needs threshold tuning for balanced recall |
| XRV Experiment | DenseNet121 (chest-X-ray-pretrained, via [TorchXRayVision](https://github.com/mlmed/torchxrayvision)) | 0.95 | 0.87 (@ default threshold 0.5) | Naturally better-calibrated confidence, no tuning needed |

Both include Grad-CAM explainability. Known limitation: the model shows some
reliance on central chest anatomy (mediastinum/heart border) rather than
purely lung tissue — this may partly reflect a real radiological sign (the
"silhouette sign," where pneumonia near the heart blurs its border) but has
not been clinically validated. See `Phase1_Radiology_Agent_Learning_Notes.docx`
for the full debugging journey (class imbalance, calibration, Grad-CAM
auditing across 6 iterations).

### Reproducing Phase 1 (Google Colab)

```python
# 1. Clone + install + get dataset
!git clone https://github.com/Zeeshan8220/clinical-vision-copilot.git
%cd clinical-vision-copilot
!pip install -r requirements.txt -q
!pip install torchxrayvision -q   # only needed for the XRV variant

from google.colab import userdata
import os
os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/pneumonia --unzip

# 2. Train the baseline model
!python src/radiology_agent/train.py --root-dir data/pneumonia/chest_xray --epochs 5

# 3. Evaluate
!python src/radiology_agent/evaluate_test.py --checkpoint checkpoints/radiology_epoch5.pt

# 4. Threshold sweep (find best decision cutoff)
!python src/radiology_agent/find_threshold.py --checkpoint checkpoints/radiology_epoch5.pt

# 5. Grad-CAM explainability (single image)
!python src/radiology_agent/gradcam.py --checkpoint checkpoints/radiology_epoch5.pt --image <path_to_xray>

# 6. Grad-CAM batch sanity-check (multiple random images)
!python src/radiology_agent/gradcam_batch.py --checkpoint checkpoints/radiology_epoch5.pt --per-class 4

# 7. Confidence calibration check
!python src/radiology_agent/calibrate.py --checkpoint checkpoints/radiology_epoch5.pt

# --- XRV variant (chest-specific pretrained backbone) ---
!python src/radiology_agent/train_xrv.py --root-dir data/pneumonia/chest_xray --epochs 5
!python src/radiology_agent/evaluate_test_xrv.py --checkpoint checkpoints_xrv/radiology_xrv_epoch5.pt
!python src/radiology_agent/find_threshold_xrv.py --checkpoint checkpoints_xrv/radiology_xrv_epoch5.pt
!python src/radiology_agent/gradcam_batch_xrv.py --checkpoint checkpoints_xrv/radiology_xrv_epoch5.pt --per-class 4
```

**Note on Colab sessions:** Google Colab's free tier resets after ~12-24
hours of inactivity or on disconnect, wiping any downloaded data (but not
this repo's code, which is safely on GitHub). If you reconnect and get
`FileNotFoundError`, just re-run step 1 above.

### Files

| File | Purpose |
|---|---|
| `dataset.py` | Data loading, preprocessing, and augmentation (both model variants) |
| `radiology_model.py` / `model_xrv.py` | Model architectures (renamed from `model.py` to avoid a naming collision with TorchXRayVision) |
| `train.py` / `train_xrv.py` | Training loops (class-weighted Focal Loss) |
| `evaluate_test.py` / `evaluate_test_xrv.py` | Test-set evaluation (AUC, precision, recall) |
| `find_threshold.py` / `find_threshold_xrv.py` | Decision-threshold tuning |
| `gradcam.py` | Single-image Grad-CAM visualization |
| `gradcam_batch.py` / `gradcam_batch_xrv.py` | Multi-image Grad-CAM sanity-check grid |
| `calibrate.py` | Temperature scaling for confidence calibration |
| `api/main.py` | FastAPI endpoint (`/health`, `/predict`) |
| `api/test_api.py` | In-process API test (no live server needed) |

## Setup (local)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scriptsctivate
pip install -r requirements.txt
```

## Current status

- [x] Repo skeleton
- [x] Phase 1 — Radiology Agent (model trained, evaluated, explainability added)
- [ ] Phase 2 — Risk Score Agent
- [ ] Phase 3 — Drug Interaction Agent (port)
- [ ] Phase 4 — Differential Diagnosis Agent (port)
- [ ] Phase 5 — Prescription Writer Agent (port)
- [ ] Phase 6 — Knowledge/RAG Agent
- [ ] Phase 7 — Orchestration
- [ ] Phase 8 — Dashboard + deploy

## Roadmap

See `Clinical_Vision_Copilot_Unified_Roadmap.docx` for the full 14-week plan
and `Phase1_Radiology_Agent_Learning_Notes.docx` for detailed Phase 1
concepts, debugging log, and lessons learned.
