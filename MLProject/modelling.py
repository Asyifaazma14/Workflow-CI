import os
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# =========================
# Config
# =========================
# Default untuk GitHub Actions/MLflow Project (dataset ada di folder MLProject)
DATA_CLEAN_PATH = os.getenv("DATA_CLEAN_PATH", "data_bersih_eksperimen.csv")
TARGET_COL = os.getenv("TARGET_COL", "Sleep Disorder")

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "Sleep_Health_Experiment_Asyifa")

# Jika kamu ingin tracking ke UI lokal, set env var:
# Windows PowerShell:
#   $env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
# CMD:
#   set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
# Kalau tidak diset, MLflow akan pakai file store default (mlruns/)
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "").strip()


def main():
    # =========================
    # MLflow setup
    # =========================
    if TRACKING_URI:
        mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Wajib: autolog (tanpa logging manual)
    mlflow.sklearn.autolog(
        log_models=True,
        log_input_examples=True,
        log_model_signatures=True
    )

    # =========================
    # Load data
    # =========================
    df = pd.read_csv(DATA_CLEAN_PATH)
    df.columns = df.columns.str.strip()

    if TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL '{TARGET_COL}' tidak ditemukan. Kolom tersedia: {df.columns.tolist()}")

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # =========================
    # Train runs (grid sederhana)
    # =========================
    param_grid = [
        {"n_estimators": 50,  "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 15},
    ]

    for params in param_grid:
        run_name = f"RF_{params['n_estimators']}_{params['max_depth']}"

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("stage", "baseline")
            mlflow.set_tag("model_type", "RandomForestClassifier")

            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42
            )

            model.fit(X_train, y_train)

            # Tidak logging manual.
            # Autolog akan menyimpan parameter + model + artifacts otomatis.

            print(f"Finished run: {run_name} | run_id={mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
