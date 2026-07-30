"""
Training script for the Risk Score Agent (XGBoost).
"""

import argparse
import joblib
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report

from dataset import load_data


def train(args):
    X_train, X_test, y_train, y_test, feature_names = load_data()

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    auc = roc_auc_score(y_test, probs)
    print(f"Test AUC: {auc:.4f}\n")
    print(classification_report(y_test, preds, target_names=["No Disease", "Heart Disease"]))

    joblib.dump(model, args.output)
    print(f"\nSaved model to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--output", default="risk_model.joblib")
    args = parser.parse_args()
    train(args)
