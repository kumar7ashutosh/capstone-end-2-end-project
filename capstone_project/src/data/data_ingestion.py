import os,numpy as np,pandas as pd,yaml
from sklearn.model_selection import train_test_split
from capstone_project.src.logger import logging

def load_params(data_path:str)->dict:
    with open(data_path,'r') as file:
        params=yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', data_path)
    return params

def load_data(data_path:str)->pd.DataFrame:
    df=pd.read_csv(data_path)
    logging.debug('dataset loaded from %s',data_path)
    return df

def save_data(df:pd.DataFrame,data_path:str):
    raw_data_path=os.path.join(data_path,'raw')
    os.makedirs(raw_data_path,exist_ok=True)
    df.to_csv(os.path.join(raw_data_path,'data.csv'),index=False)
    logging.debug('dataset uploaded at %s',raw_data_path)
    
def main():
    df=load_data('https://raw.githubusercontent.com/kumar7ashutosh/credit_card/refs/heads/main/data.csv')
    save_data(df,'capstone_project/data')

if __name__=='__main__':
    main()
    