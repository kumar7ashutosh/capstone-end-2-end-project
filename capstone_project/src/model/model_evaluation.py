import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import os
import sys
from capstone_project.src.logger import logging

import os
import mlflow

os.environ["MLFLOW_TRACKING_USERNAME"] = "kumarashutoshbtech2023"
token = os.getenv("DAGSHUB_TOKEN")
if token:
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
else:
    print("⚠ DAGSHUB_TOKEN not found — skipping MLflow logging.")
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/kumarashutoshbtech2023/capstone-end-2-end-project.mlflow"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
def load_data(data_path:str)->pd.DataFrame:
    df=pd.read_csv(data_path)
    logging.debug('data retrieved from %s', data_path)
    return df

def load_model(file_path:str):
    with open(file_path,'rb') as file:
        model=pickle.load(file)
        logging.info('model loaded from %s', file_path)
    return model

def evaluate_model(clf,x_test:np.ndarray,y_test:np.ndarray)->dict:
    y_pred=clf.predict(x_test)
    y_pred_proba=clf.predict_proba(x_test)[:,1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
    logging.info('model evaluation metrics calculated')
    return metrics_dict

def save_metrics(metrics:dict,file_path:str):
    with open(file_path,'w') as file:
        json.dump(metrics,file,indent=4)
    logging.info('Metrics saved to %s', file_path)
    
def save_model_info(run_id:str,model_path:str,file_path:str):
    model_info={'run_id':run_id,
                'model_path':model_path}
    with open(file_path,'w') as file:
        json.dump(model_info,file,indent=4)
    logging.debug('Model info saved to %s', file_path)
    
def main():
    mlflow.set_experiment('dvc-capstone-pipeline')
    with mlflow.start_run() as run:
        clf=load_model('capstone_project/models/model.pkl')
        test_df=load_data('capstone_project/data/processed/transformed_test.csv')
        x_test=test_df.iloc[:,:-1].values
        y_test=test_df.iloc[:,-1].values
        metrics=evaluate_model(clf=clf,x_test=x_test,y_test=y_test)
        save_metrics(metrics,'capstone_project/reports/metrics.json')
        for metric_name,metric_value in metrics.items():
            mlflow.log_metric(metric_name,metric_value)
        if hasattr(clf,'get_params'):
            params=clf.get_params()
            for param_name,param_value in params.items():
                mlflow.log_param(param_name,param_value)
        
        save_model_info(run.info.run_id,'model','capstone_project/reports/experiment_info.json')
        mlflow.log_artifact('capstone_project/reports/metrics.json')
        
if __name__=='__main__':
    main()
        
 