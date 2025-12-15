import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_CLEAN_PATH = "data_bersih_automasi.csv"
TARGET_COL = "Sleep Disorder"


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average="weighted", zero_division=0)
    recall = recall_score(actual, pred, average="weighted", zero_division=0)
    f1 = f1_score(actual, pred, average="weighted", zero_division=0)
    return accuracy, precision, recall, f1


def main():
    # Load data
    df = pd.read_csv(DATA_CLEAN_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL '{TARGET_COL}' tidak ditemukan. Kolom tersedia: {df.columns.tolist()}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

 
    # MLflow setup (BASELINE -> localhost)
    EXPERIMENT_NAME = "Sleep_Health_Experiment_Asyifa"

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Wajib: autolog (sklearn) dan TIDAK logging manual
    mlflow.sklearn.autolog(
        log_models=True,
        log_input_examples=True,
        log_model_signatures=True
    )

 
    # Train + run per setting
    param_grid = [
        {"n_estimators": 50,  "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 15},
    ]

    best_f1 = -1.0
    best_run = None

    for params in param_grid:
        run_name = f"RF_{params['n_estimators']}_{params['max_depth']}"

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("stage", "baseline")
            mlflow.set_tag("model_type", "RandomForestClassifier")

            rf = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42
            )

            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            acc, prec, rec, f1 = eval_metrics(y_test, y_pred)

            # Tidak logging manual 
            print(f"Run: {run_name}")
            print(f"  acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
            print(f"  Run ID: {mlflow.active_run().info.run_id}")
            print("-" * 40)

            if f1 > best_f1:
                best_f1 = f1
                best_run = run_name

    print(f"Best F1: {best_f1:.4f} (Run: {best_run})")


if __name__ == "__main__":
    main()
