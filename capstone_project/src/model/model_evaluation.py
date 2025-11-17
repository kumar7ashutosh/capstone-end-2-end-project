import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import os
import mlflow

# Remove inherited GitHub Actions environment overrides
for key in ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]:
    if key in os.environ:
        print(f"🔧 Removing inherited env var: {key}")
        os.environ.pop(key)

DAGSHUB_USER = "kumarashutoshbtech2023"
DAGSHUB_REPO = "capstone-end-2-end-project"

token = os.getenv("DAGSHUB_TOKEN")

if token:
    print("🔐 DAGSHUB_TOKEN detected — using DagsHub MLflow")
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")
else:
    print("⚠ DAGSHUB_TOKEN not found — using LOCAL MLflow instead")

    # Fully reset MLflow
    mlflow.set_tracking_uri(None)
    mlflow.set_tracking_uri("file:./mlruns")

# ======================================
# UTILITIES
# ======================================
def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    logging.debug("data retrieved from %s", data_path)
    return df


def load_model(file_path: str):
    with open(file_path, "rb") as file:
        model = pickle.load(file)
        logging.info("model loaded from %s", file_path)
    return model


def evaluate_model(clf, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = clf.predict(x_test)
    y_pred_proba = clf.predict_proba(x_test)[:, 1]

    metrics_dict = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred_proba),
    }

    logging.info("model evaluation metrics calculated")
    return metrics_dict


def save_metrics(metrics: dict, file_path: str):
    with open(file_path, "w") as file:
        json.dump(metrics, file, indent=4)
    logging.info("Metrics saved to %s", file_path)


def save_model_info(run_id: str, model_path: str, file_path: str):
    model_info = {"run_id": run_id, "model_path": model_path}
    with open(file_path, "w") as file:
        json.dump(model_info, file, indent=4)
    logging.debug("Model info saved to %s", file_path)


# ======================================
# MAIN
# ======================================
def main():
    mlflow.set_experiment("dvc-capstone-pipeline")

    with mlflow.start_run() as run:
        clf = load_model("capstone_project/models/model.pkl")
        test_df = load_data("capstone_project/data/processed/transformed_test.csv")

        x_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values

        metrics = evaluate_model(clf=clf, x_test=x_test, y_test=y_test)

        save_metrics(metrics, "capstone_project/reports/metrics.json")

        # Log metrics
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # Log model params
        if hasattr(clf, "get_params"):
            params = clf.get_params()
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

        # Save run info
        save_model_info(
            run.info.run_id,
            "model",
            "capstone_project/reports/experiment_info.json",
        )

        mlflow.log_artifact("capstone_project/reports/metrics.json")


if __name__ == "__main__":
    main()
