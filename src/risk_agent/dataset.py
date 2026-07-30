"""
Dataset loader for the Risk Score Agent.
Uses the Kaggle "Heart Failure Prediction" dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

CATEGORICAL_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
TARGET_COL = "HeartDisease"


def load_data(csv_path="data/heart/heart.csv", test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS)

    X = df_encoded.drop(columns=[TARGET_COL])
    y = df_encoded[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, X.columns.tolist()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Train label balance:\n{y_train.value_counts()}")
