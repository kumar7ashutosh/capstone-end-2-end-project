import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
import os
import sys
from capstone_project.src.logger import logging
from sklearn.preprocessing import PowerTransformer
def load_params(file_path:str)->dict:
    with open(file_path,'r') as file:
        params=yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', file_path)
    return params

def load_data(data_path:str)->pd.DataFrame:
    df=pd.read_csv(data_path)
    logging.debug('data retrieved from %s', data_path)
    return df

def train_model(x_train:np.ndarray,y_train:np.ndarray)->LogisticRegression:
    params=load_params('params.yaml')
    C=params['model']['C']
    solver=params['model']['solver']
    penalty=params['model']['penalty']
    clf=LogisticRegression(C=C,solver=solver,penalty=penalty)
    clf.fit(x_train,y_train)
    logging.info('model trained')
    return clf
def save_transformer(transformer,file_path:str)->None:
    with open(file_path,'wb') as file:
        pickle.dump(transformer,file)
        logging.info('PowerTransformer saved to %s', file_path)

def save_model(model,file_path:str)->None:
    with open(file_path,'wb') as file:
        pickle.dump(model,file)
        logging.info('model saved to %s',file_path)
        
def main():
    train_df=load_data('capstone_project/data/processed/transformed_train.csv')
    x_train=train_df.iloc[:,:-1].values
    y_train=train_df.iloc[:,-1].values
    power_transformer=PowerTransformer(method='yeo-johnson')
    x_train_transformed=power_transformer.fit_transform(x_train)
    clf=train_model(x_train=x_train_transformed,y_train=y_train)
    save_transformer(power_transformer,'capstone_project/models/power_transformer.pkl')
    save_transformer(clf,'capstone_project/models/model.pkl')

if __name__=='__main__':
    main()   