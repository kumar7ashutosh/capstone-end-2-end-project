import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import dagshub
from dotenv import load_dotenv  # new

# Load environment variables from .env file if present
load_dotenv()

# Set DagsHub token from environment
DAGSHUB_TOKEN = os.environ.get("DAGSHUB_TOKEN")

# Initialize MLflow/DagsHub only if token exists
if DAGSHUB_TOKEN:
    os.environ["DAGSHUB_TOKEN"] = DAGSHUB_TOKEN
    mlflow.set_tracking_uri('https://dagshub.com/kumarashutoshbtech2023/capstone-end-2-end-project.mlflow')
    dagshub.init(
        repo_owner='kumarashutoshbtech2023',
        repo_name='capstone-end-2-end-project',
        mlflow=True
    )
    logging.info("MLflow/DagsHub initialized")
else:
    logging.warning("DAGSHUB_TOKEN not found. MLflow logging will use local SQLite.")
    mlflow.set_tracking_uri('sqlite:///mlflow.db')

# Paths (can be made configurable)
MODEL_PATH = 'capstone_project/models/model.pkl'
TEST_DATA_PATH = 'capstone_project/data/processed/transformed_test.csv'
METRICS_PATH = 'capstone_project/reports/metrics.json'
EXPERIMENT_INFO_PATH = 'capstone_project/reports/experiment_info.json'

# Load data
def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    logging.debug(f"Data loaded from {data_path}")
    return df

# Load model
def load_model(file_path: str):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
        logging.info(f"Model loaded from {file_path}")
    return model

# Evaluate model
def evaluate_model(clf, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = clf.predict(x_test)
    y_pred_proba = clf.predict_proba(x_test)[:, 1]
    metrics_dict = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    logging.info("Model evaluation metrics calculated")
    return metrics_dict

# Save metrics to JSON
def save_metrics(metrics: dict, file_path: str):
    with open(file_path, 'w') as file:
        json.dump(metrics, file, indent=4)
    logging.info(f"Metrics saved to {file_path}")

# Save experiment info
def save_model_info(run_id: str, model_path: str, file_path: str):
    model_info = {'run_id': run_id, 'model_path': model_path}
    with open(file_path, 'w') as file:
        json.dump(model_info, file, indent=4)
    logging.debug(f"Model info saved to {file_path}")

# Main function
def main():
    mlflow.set_experiment('dvc-capstone-pipeline')
    with mlflow.start_run() as run:
        clf = load_model(MODEL_PATH)
        test_df = load_data(TEST_DATA_PATH)
        x_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values

        metrics = evaluate_model(clf, x_test, y_test)
        save_metrics(metrics, METRICS_PATH)
        mlflow.log_metrics(metrics)  # log all metrics at once

        # Log model parameters
        if hasattr(clf, 'get_params'):
            mlflow.log_params(clf.get_params())

        save_model_info(run.info.run_id, MODEL_PATH, EXPERIMENT_INFO_PATH)
        mlflow.log_artifact(METRICS_PATH)

if __name__ == "__main__":
    main()
