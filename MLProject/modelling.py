import os
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


DATA_CLEAN_PATH = os.getenv("DATA_CLEAN_PATH", "data_bersih_eksperimen.csv")
TARGET_COL = os.getenv("TARGET_COL", "Sleep Disorder")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "Sleep_Health_Experiment_Asyifa")

def main():
    # tracking uri optional (kalau diset lewat env)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # wajib autolog (tanpa logging manual)
    mlflow.sklearn.autolog(
        log_models=True,
        log_input_examples=True,
        log_model_signatures=True
    )

    df = pd.read_csv(DATA_CLEAN_PATH)
    df.columns = df.columns.str.strip()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1 model saja agar aman di MLflow Projects CI
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Training done.")

if __name__ == "__main__":
    main()
