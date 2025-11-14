import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from dotenv import load_dotenv
load_dotenv()

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

MLFLOW_TRACKING_URI = "https://dagshub.com/kumarashutoshbtech2023/capstone-end-2-end-project.mlflow"
repo_owner = "kumarashutoshbtech2023"
repo_name = "capstone-end-2-end-project"

dagshub.init(repo_owner=repo_owner,repo_name=repo_name,mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment('logistic regression with powertransformer')

def load_preprocess_data(filepath):
    df=pd.read_csv(filepath)
    x=df.drop(columns=['Class'],axis=1)
    y=df['Class']
    transformer=PowerTransformer('yeo-johnson')
    x_transformed=transformer.fit_transform(x)
    
    return train_test_split(x_transformed, y, test_size=0.2, random_state=42), transformer

def train_and_log_model(x_train,x_test,y_train,y_test,transformer):
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }
    with mlflow.start_run():
        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid_search.fit(x_train, y_train)

        # Log all hyperparameter tuning runs
        for params, mean_score, std_score in zip(grid_search.cv_results_["params"], 
                                                 grid_search.cv_results_["mean_test_score"], 
                                                 grid_search.cv_results_["std_test_score"]):
            with mlflow.start_run(run_name=f"LR with params: {params}", nested=True):
                model = LogisticRegression(**params)
                model.fit(x_train, y_train)
                
                y_pred = model.predict(x_test)
                
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score
                }
                
                # Log parameters & metrics
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                
                print(f"Params: {params} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

        # Log the best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_f1)
        
        print(f"\nBest Params: {best_params} | Best F1 Score: {best_f1:.4f}")
if __name__=="__main__":
    (x_train,x_test,y_train,y_test),transformer=load_preprocess_data('capstone_project/notebooks/data.csv')
    train_and_log_model(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,transformer=transformer)