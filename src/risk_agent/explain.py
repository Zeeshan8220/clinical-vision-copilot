"""
SHAP explainability for the Risk Score Agent.
"""

import joblib
import shap
import matplotlib.pyplot as plt

from dataset import load_data


def main():
    X_train, X_test, y_train, y_test, feature_names = load_data()
    model = joblib.load("risk_model.joblib")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=130, bbox_inches="tight")
    print("Saved global feature importance to shap_summary.png")

    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig("shap_waterfall_patient0.png", dpi=130, bbox_inches="tight")
    print("Saved single-patient explanation to shap_waterfall_patient0.png")

    pred_prob = model.predict_proba(X_test.iloc[[0]])[0, 1]
    print(f"\nPatient 0: predicted {pred_prob:.1%} probability of heart disease")
    print(f"Actual label: {'Heart Disease' if y_test.iloc[0] == 1 else 'No Disease'}")


if __name__ == "__main__":
    main()
