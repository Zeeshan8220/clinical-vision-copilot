"""
Clinical Vision Copilot -- Streamlit dashboard.
"""

import streamlit as st
import sys
import os
import tempfile

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "orchestrator"))

# Bridge Streamlit secrets to environment variables (agents read via os.environ)
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

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

    st.success("✅ Analysis complete")
    st.divider()

    rad = report["radiology"]
    st.subheader("🫁 Radiology")
    if rad.get("skipped"):
        st.info("No X-ray uploaded — radiology analysis skipped.")
    else:
        color = "🔴" if rad["prediction"] == "PNEUMONIA" else "🟢"
        st.markdown(f"**{color} Prediction: {rad['prediction']}**  (confidence: {rad['confidence']:.0%})")

    st.divider()

    risk = report["risk_score"]
    st.subheader("❤️ Cardiac Risk Score")
    if risk.get("skipped"):
        st.info("No clinical data provided — risk scoring skipped.")
    else:
        level_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(risk["risk_level"], "⚪")
        col_a, col_b = st.columns(2)
        col_a.metric("Risk Probability", f"{risk['risk_probability']:.1%}")
        col_b.metric("Risk Level", f"{level_color} {risk['risk_level']}")

    st.divider()

    drugs = report["drug_interactions"]
    st.subheader("💊 Drug Interactions")
    if drugs.get("skipped"):
        st.info("No medications provided — interaction check skipped.")
    else:
        interactions = drugs.get("interactions", [])
        if not interactions:
            st.success("No known interactions found among the listed medications.")
        else:
            severity_color = {"Contraindicated": "🔴", "Major": "🟠", "Moderate": "🟡", "Minor": "🟢", "Unknown": "⚪"}
            for i in interactions:
                badge = severity_color.get(i["severity"], "⚪")
                source_tag = "✅ Verified" if i["source"] == "verified_database" else "🤖 AI-generated (unverified)"
                st.markdown(f"{badge} **{i['drug_a'].title()} + {i['drug_b'].title()}** — *{i['severity']}* ({source_tag})")
                st.caption(i["description"])

    st.divider()

    dx = report["differential_diagnosis"]
    st.subheader("🩺 Differential Diagnosis")
    if dx.get("skipped"):
        st.info("No symptoms provided — differential diagnosis skipped.")
    else:
        likelihood_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
        for item in dx.get("differential", []):
            badge = likelihood_color.get(item["likelihood"], "⚪")
            with st.expander(f"{badge} {item['diagnosis']} — {item['likelihood']} likelihood"):
                st.write(item["reasoning"])

    st.divider()

    rx = report["prescription_draft"]
    st.subheader("📝 Prescription Draft")
    if rx.get("skipped"):
        st.info("No diagnosis available — prescription draft skipped.")
    else:
        for med in rx.get("medications", []):
            st.markdown(f"**{med['name']}** — {med['dosage']}, {med['frequency']}, for {med['duration']}")
            st.caption(med.get("notes", ""))
        st.markdown(f"**Follow-up:** {rx.get('follow_up', 'N/A')}")
        st.markdown(f"**Referral:** {rx.get('referral', 'N/A')}")
        st.markdown(f"**🚨 Red flags:** {rx.get('red_flags', 'N/A')}")

    st.divider()

    kb = report["knowledge_reference"]
    st.subheader("📚 Knowledge Reference")
    if kb.get("skipped"):
        st.info("No diagnosis to look up.")
    else:
        st.write(kb.get("answer", ""))
        sources = list(set(kb.get("retrieved_sources", [])))
        if sources:
            st.caption("Sources: " + ", ".join(sources))

    st.divider()
    st.warning(report["disclaimer"])
