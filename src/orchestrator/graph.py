"""
Orchestrator -- chains all 6 agents together (via LangGraph) so a
single patient case produces one combined, explainable report.
"""

import sys
import os
import importlib.util

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO_ROOT, "src")

def load_module(unique_name, file_path):
    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module

radiology_dataset = load_module("radiology_dataset", os.path.join(SRC, "radiology_agent", "dataset.py"))
radiology_model_mod = load_module("radiology_model_mod", os.path.join(SRC, "radiology_agent", "radiology_model.py"))
risk_dataset = load_module("risk_dataset", os.path.join(SRC, "risk_agent", "dataset.py"))

sys.path.append(os.path.join(SRC, "drug_interaction_agent"))
sys.path.append(os.path.join(SRC, "differential_dx_agent"))
sys.path.append(os.path.join(SRC, "prescription_agent"))
sys.path.append(os.path.join(SRC, "knowledge_agent"))

from interaction_checker import check_interactions_hybrid
from differential_dx import get_differential_diagnosis
from prescription_writer import generate_prescription_draft
from rag_agent import answer_question

import torch
import joblib
import pandas as pd
from PIL import Image
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END


class CaseState(TypedDict, total=False):
    xray_image_path: Optional[str]
    patient_clinical: Optional[dict]
    medications: Optional[List[str]]
    symptoms: Optional[str]
    age: Optional[int]
    sex: Optional[str]
    radiology_result: dict
    risk_result: dict
    drug_result: dict
    differential_result: dict
    prescription_result: dict
    knowledge_result: dict
    final_report: dict


def radiology_node(state: CaseState) -> CaseState:
    path = state.get("xray_image_path")
    if not path:
        state["radiology_result"] = {"skipped": True, "reason": "no image provided"}
        return state

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = radiology_model_mod.RadiologyClassifier(num_classes=len(radiology_dataset.CLASS_NAMES))
    checkpoint_path = os.path.join(REPO_ROOT, "checkpoints", "radiology_epoch5.pt")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    image = Image.open(path).convert("L")
    input_tensor = radiology_dataset.EVAL_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)[0]
        pneumonia_prob = float(probs[1])

    threshold = 0.8
    prediction = "PNEUMONIA" if pneumonia_prob >= threshold else "NORMAL"
    confidence = pneumonia_prob if prediction == "PNEUMONIA" else (1 - pneumonia_prob)

    state["radiology_result"] = {"prediction": prediction, "confidence": round(confidence, 4)}
    return state


def risk_node(state: CaseState) -> CaseState:
    clinical = state.get("patient_clinical")
    if not clinical:
        state["risk_result"] = {"skipped": True, "reason": "no clinical data provided"}
        return state

    model = joblib.load(os.path.join(REPO_ROOT, "risk_model.joblib"))
    train_columns = model.get_booster().feature_names

    row = pd.DataFrame([clinical])
    row_encoded = pd.get_dummies(row, columns=risk_dataset.CATEGORICAL_COLS)
    row_aligned = row_encoded.reindex(columns=train_columns, fill_value=0)

    prob = float(model.predict_proba(row_aligned)[0, 1])
    risk_level = "High" if prob >= 0.6 else ("Medium" if prob >= 0.3 else "Low")
    state["risk_result"] = {"risk_probability": round(prob, 4), "risk_level": risk_level}
    return state


def drug_node(state: CaseState) -> CaseState:
    meds = state.get("medications")
    if not meds:
        state["drug_result"] = {"skipped": True, "reason": "no medications provided"}
        return state
    state["drug_result"] = {"interactions": check_interactions_hybrid(meds, use_llm=True)}
    return state


def dx_node(state: CaseState) -> CaseState:
    symptoms = state.get("symptoms")
    if not symptoms:
        state["differential_result"] = {"skipped": True, "reason": "no symptoms provided"}
        return state
    result = get_differential_diagnosis(symptoms, age=state.get("age"), sex=state.get("sex"))
    state["differential_result"] = result
    return state


def prescription_node(state: CaseState) -> CaseState:
    dx = state.get("differential_result", {})
    diffs = dx.get("differential", [])
    if not diffs:
        state["prescription_result"] = {"skipped": True, "reason": "no diagnosis available to prescribe for"}
        return state
    top_diagnosis = diffs[0]["diagnosis"]
    result = generate_prescription_draft(diagnosis=top_diagnosis, age=state.get("age"), sex=state.get("sex"))
    state["prescription_result"] = result
    return state


def knowledge_node(state: CaseState) -> CaseState:
    dx = state.get("differential_result", {})
    diffs = dx.get("differential", [])
    if not diffs:
        state["knowledge_result"] = {"skipped": True, "reason": "no diagnosis to look up"}
        return state
    top_diagnosis = diffs[0]["diagnosis"]
    question = f"What is the recommended management or treatment for {top_diagnosis}?"
    state["knowledge_result"] = answer_question(question)
    return state


def combine_node(state: CaseState) -> CaseState:
    state["final_report"] = {
        "radiology": state.get("radiology_result"),
        "risk_score": state.get("risk_result"),
        "drug_interactions": state.get("drug_result"),
        "differential_diagnosis": state.get("differential_result"),
        "prescription_draft": state.get("prescription_result"),
        "knowledge_reference": state.get("knowledge_result"),
        "disclaimer": (
            "This is a decision-support / educational tool combining multiple "
            "AI agents. It is NOT a diagnostic medical device and does NOT "
            "replace clinical judgement. Always consult a licensed physician."
        ),
    }
    return state


def build_graph():
    graph = StateGraph(CaseState)
    graph.add_node("radiology", radiology_node)
    graph.add_node("risk", risk_node)
    graph.add_node("drugs", drug_node)
    graph.add_node("differential", dx_node)
    graph.add_node("prescription", prescription_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("combine", combine_node)

    graph.set_entry_point("radiology")
    graph.add_edge("radiology", "risk")
    graph.add_edge("risk", "drugs")
    graph.add_edge("drugs", "differential")
    graph.add_edge("differential", "prescription")
    graph.add_edge("prescription", "knowledge")
    graph.add_edge("knowledge", "combine")
    graph.add_edge("combine", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    case_input = {
        "symptoms": "fever, cough, chest pain, shortness of breath for 3 days",
        "age": 45,
        "sex": "M",
        "medications": ["Warfarin", "Ibuprofen"],
        "patient_clinical": {
            "Age": 45, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140,
            "Cholesterol": 250, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 130,
            "ExerciseAngina": "Y", "Oldpeak": 1.5, "ST_Slope": "Flat",
        },
    }

    result = app.invoke(case_input)
    import json
    print(json.dumps(result["final_report"], indent=2))
