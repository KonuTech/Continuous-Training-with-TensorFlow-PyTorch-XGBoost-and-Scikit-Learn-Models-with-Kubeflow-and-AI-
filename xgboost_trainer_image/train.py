
import os 
import subprocess
import datetime
import fire
import pickle 

import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def train_evaluate(training_dataset_path, validation_dataset_path,max_depth,n_estimators,output_dir):
    
    df_train = pd.read_csv(training_dataset_path)
    df_validation = pd.read_csv(validation_dataset_path)
    df = pd.concat([df_train, df_validation])

    categorical_features = ['workclass', 'occupation']
    target='income_bracket'

    # One-hot encode categorical variables 
    df = pd.get_dummies(df,columns=categorical_features)

    # Change label to 0 if <=50K, 1 if >50K
    df[target] = df[target].apply(lambda x: 0 if x==' <=50K' else 1)

    # Split features and labels into 2 different vars
    X_train = df.loc[:, df.columns != target]
    y_train = np.array(df[target])

    # Normalize features 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    grid = {
        'max_depth': int(max_depth),
        'n_estimators': int(n_estimators)
    }
    
    model = XGBClassifier()
    model.set_params(**grid)
    model.fit(X_train,y_train)
    
    model_filename = 'xgb_model.pkl'
    pickle.dump(model, open(model_filename, "wb"))
        
    EXPORT_PATH = os.path.join(
        output_dir, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    
    gcs_model_path = '{}/{}'.format(EXPORT_PATH, model_filename)
    subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path])
    print('Saved model in: {}'.format(gcs_model_path))  

if __name__ == '__main__':
    fire.Fire(train_evaluate)
