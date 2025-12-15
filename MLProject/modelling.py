import os
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ========== CONFIG ==========
DATA_CLEAN_PATH = os.getenv("DATA_CLEAN_PATH", "data_bersih_eksperimen.csv")
TARGET_COL = os.getenv("TARGET_COL", "Sleep Disorder")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "Sleep_Health_Experiment_Asyifa")


def main():
    # (Opsional) kalau mau tracking ke server tertentu via env var
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Wajib: autolog (tanpa logging manual)
    mlflow.sklearn.autolog(
        log_models=True,
        log_input_examples=True,
        log_model_signatures=True
    )

    # ========== LOAD DATA ==========
    df = pd.read_csv(DATA_CLEAN_PATH)
    df.columns = df.columns.str.strip()

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"TARGET_COL '{TARGET_COL}' tidak ditemukan. Kolom tersedia: {df.columns.tolist()}"
        )

    # ========== PREPROCESS FIX (STRING -> NUMERIC) ==========
    # Blood Pressure: "125/80" -> BP_Systolic, BP_Diastolic
    bp_col = "Blood Pressure"
    if bp_col in df.columns:
        bp_split = df[bp_col].astype(str).str.split("/", expand=True)
        if bp_split.shape[1] == 2:
            df["BP_Systolic"] = pd.to_numeric(bp_split[0], errors="coerce")
            df["BP_Diastolic"] = pd.to_numeric(bp_split[1], errors="coerce")
            df = df.drop(columns=[bp_col])

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # One-hot encode categorical features (biar model bisa fit)
    X_raw = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X = pd.get_dummies(X_raw, drop_first=True)

    # Pastikan semua numerik (kalau masih ada yang nyelip jadi object)
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Buang baris yang masih NaN setelah coercion
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # ========== SPLIT ==========
    # stratify aman kalau y punya >1 class
    stratify_y = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_y
    )

    # ========== TRAIN ==========
    # 1 model saja (aman untuk CI + MLflow Projects)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Training done (MLflow autolog will capture params + model artifacts).")


if __name__ == "__main__":
    main()
