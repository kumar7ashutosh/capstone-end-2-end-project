import os,numpy as np,pandas as pd,yaml
from capstone_project.src.logger import logging
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

def load_params(params_path:str)->dict:
    with open(params_path,'r') as file:
        params=yaml.safe_load(file)
        logging.info('params loaded from %s',params_path)
    return params

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    if 'Class' in df.columns:
        normal=df[df['Class']==0]
        fraud=df[df['Class']==1]
        logging.info(f'normal shape {normal.shape}, fraud shape {fraud.shape}')
        normal_under_sample=resample(normal,replace=False,n_samples=len(fraud),random_state=27)
        logging.info(f'normal shape {normal.shape}, fraud shape {fraud.shape} after sampling')
        new_df=pd.concat([normal_under_sample,fraud])
        logging.info(f'resampled dataset {new_df.shape}')
        return new_df

def main():
    params=load_params('params.yaml')
    test_size=params['data_preprocessing']['test_size']
    df=pd.read_csv(r'capstone_project\data\raw\data.csv')
    new_df=preprocess_data(df=df)
    train_df,test_df=train_test_split(new_df,stratify=new_df['Class'],test_size=test_size,random_state=42)
    logging.info(f'train data shape {train_df.shape} , test data shape {test_df.shape}')
    preprocess_data_path=os.path.join('capstone_project/data','interim')
    os.makedirs(preprocess_data_path,exist_ok=True)
    train_df.to_csv(os.path.join(preprocess_data_path,'train_df.csv'),index=False)    
    test_df.to_csv(os.path.join(preprocess_data_path,'test_df.csv'),index=False)    
    logging.info('train and test data saved at %s',preprocess_data_path)

if __name__=='__main__':
    main()
