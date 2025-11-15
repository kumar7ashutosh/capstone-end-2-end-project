import numpy as np
import pandas as pd
import os
import sys
import logging
from sklearn.preprocessing import PowerTransformer
from capstone_project.src.logger import logging

def load_data(data_path:str)->pd.DataFrame:
    df=pd.read_csv(data_path)
    logging.debug('dataset loaded from %s',data_path)
    return df

def feature_engineering(train_df:pd.DataFrame,test_df:pd.DataFrame)->tuple:
    x_train=train_df.drop(columns=['Class'])
    y_train=train_df['Class']
    x_test=test_df.drop(columns=['Class'])
    y_test=test_df['Class']
    num_columns=x_train.select_dtypes(include=['number']).columns.tolist()
    pt=PowerTransformer(method='yeo-johnson')
    x_train[num_columns]=pt.fit_transform(x_train[num_columns])
    x_test[num_columns]=pt.transform(x_test[num_columns])
    train_transformed=pd.concat([x_train,y_train],axis=1)
    test_transformed=pd.concat([x_test,y_test],axis=1)
    logging.info('train and test data reconstructed using powertransformer')
    return train_transformed,test_transformed

def main():
    train_df=load_data('capstone_project/data/interim/train_df.csv')
    test_df=load_data('capstone_project/data/interim/test_df.csv')
    transformed_train_df,transformed_test_df=feature_engineering(train_df=train_df,test_df=test_df)
    transformed_path=os.path.join('capstone_project/data','processed')
    os.makedirs(transformed_path,exist_ok=True)
    transformed_train_df.to_csv(os.path.join(transformed_path,'transformed_train.csv'),index=False)
    transformed_test_df.to_csv(os.path.join(transformed_path,'transformed_test.csv'),index=False)
    logging.info('transformed training data is saved at %s',transformed_path)
    logging.info('transformed testing data is saved at %s',transformed_path)

if __name__=='__main__':
    main()