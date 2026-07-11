# Clinical Vision Copilot

Multi-agent AI decision-support system for medical case analysis. Built solo,
zero-budget, using free tools and public datasets only.

**This is a decision-support / educational tool. It is not a diagnostic
medical device and is not a substitute for a licensed doctor's judgement.**

## Architecture

Six agents, orchestrated with LangGraph:

| # | Agent | Status | Folder |
|---|-------|--------|--------|
| 1 | Radiology Agent | Build (Phase 1) | `src/radiology_agent/` |
| 2 | Risk Score Agent | Build (Phase 2) | `src/risk_agent/` |
| 3 | Drug Interaction Agent | Port from MedGenius (Phase 3) | `src/drug_interaction_agent/` |
| 4 | Differential Diagnosis Agent | Port from MedGenius (Phase 4) | `src/differential_dx_agent/` |
| 5 | Prescription Writer Agent | Port from MedGenius (Phase 5) | `src/prescription_agent/` |
| 6 | Knowledge/RAG Agent | Build (Phase 6) | `src/knowledge_agent/` |
| — | Orchestrator | Phase 7 | `src/orchestrator/` |

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Current status

- [x] Repo skeleton
- [ ] Phase 1 — Radiology Agent
- [ ] Phase 2 — Risk Score Agent
- [ ] Phase 3 — Drug Interaction Agent (port)
- [ ] Phase 4 — Differential Diagnosis Agent (port)
- [ ] Phase 5 — Prescription Writer Agent (port)
- [ ] Phase 6 — Knowledge/RAG Agent
- [ ] Phase 7 — Orchestration
- [ ] Phase 8 — Dashboard + deploy

## Dataset (Phase 1)

Using the [ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) subset
to start (smaller and faster to iterate on than full LIDC-IDRI CT volumes).
Download instructions are in `src/radiology_agent/dataset.py`.

## Roadmap

See `Clinical_Vision_Copilot_Unified_Roadmap.docx` for the full 14-week plan.
