"""
Clinical Vision Copilot -- Streamlit dashboard.
"""

import streamlit as st
import sys
import os
import tempfile

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "orchestrator"))
from graph import build_graph

st.set_page_config(page_title="Clinical Vision Copilot", layout="wide")

st.title("🏥 Clinical Vision Copilot")
st.caption(
    "A multi-agent AI decision-support demo -- radiology, risk scoring, "
    "drug interactions, differential diagnosis, prescription drafting, "
    "and guideline lookup, combined into one report."
)
st.warning(
    "⚠️ This is a decision-support / educational tool, NOT a diagnostic "
    "medical device. It does not replace clinical judgement."
)

with st.form("case_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Patient & Symptoms")
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        sex = st.selectbox("Sex", ["M", "F"])
        symptoms = st.text_area(
            "Symptoms",
            "fever, cough, chest pain, shortness of breath for 3 days",
        )
        medications = st.text_input(
            "Current medications (comma-separated)", "Warfarin, Ibuprofen"
        )
        xray_file = st.file_uploader("Chest X-ray (optional)", type=["jpg", "jpeg", "png"])

    with col2:
        st.markdown("### Clinical Data (for Risk Score)")
        chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
        resting_bp = st.number_input("Resting BP", value=140)
        cholesterol = st.number_input("Cholesterol", value=250)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.number_input("Max Heart Rate", value=130)
        exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
        oldpeak = st.number_input("Oldpeak", value=1.5)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("Run Analysis")

if submitted:
    with st.spinner("Running all agents..."):
        image_path = None
        if xray_file is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tmp.write(xray_file.read())
            tmp.close()
            image_path = tmp.name

        case_input = {
            "symptoms": symptoms,
            "age": int(age),
            "sex": sex,
            "medications": [m.strip() for m in medications.split(",") if m.strip()],
            "xray_image_path": image_path,
            "patient_clinical": {
                "Age": int(age), "Sex": sex, "ChestPainType": chest_pain,
                "RestingBP": int(resting_bp), "Cholesterol": int(cholesterol),
                "FastingBS": int(fasting_bs), "RestingECG": resting_ecg,
                "MaxHR": int(max_hr), "ExerciseAngina": exercise_angina,
                "Oldpeak": float(oldpeak), "ST_Slope": st_slope,
            },
        }

        app_graph = build_graph()
        result = app_graph.invoke(case_input)
        report = result["final_report"]

    st.success("Analysis complete")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🫁 Radiology")
        st.json(report["radiology"])
    with c2:
        st.subheader("❤️ Risk Score")
        st.json(report["risk_score"])
    with c3:
        st.subheader("💊 Drug Interactions")
        st.json(report["drug_interactions"])

    st.subheader("🩺 Differential Diagnosis")
    st.json(report["differential_diagnosis"])

    st.subheader("📝 Prescription Draft")
    st.json(report["prescription_draft"])

    st.subheader("📚 Knowledge Reference")
    st.json(report["knowledge_reference"])

    st.warning(report["disclaimer"])
