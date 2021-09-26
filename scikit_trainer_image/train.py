
"""Census Scikit-learn classifier trainer script."""

import pickle
import subprocess
import sys
import datetime
import os

import fire
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def train_evaluate(training_dataset_path, validation_dataset_path,output_dir):
    """Trains the Census Classifier model."""
    
    # Ingest data into Pandas Dataframes 
    df_train = pd.read_csv(training_dataset_path)
    df_validation = pd.read_csv(validation_dataset_path)
    df_train = pd.concat([df_train, df_validation])
    
    numeric_features = [
        'age', 'education_num','hours_per_week'
    ]
    
    categorical_features = ['workclass', 'occupation']
    
    # Scale numeric features, one-hot encode categorical features
    preprocessor = ColumnTransformer(transformers=[(
        'num', StandardScaler(),
        numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])
    
    pipeline = Pipeline([('preprocessor', preprocessor),
                         ('classifier', SGDClassifier(loss='log'))])
    
    num_features_type_map = {feature: 'float64' for feature in numeric_features}
    df_train = df_train.astype(num_features_type_map)
    df_validation = df_validation.astype(num_features_type_map)
    
    X_train = df_train.drop('income_bracket', axis=1)
    y_train = df_train['income_bracket']
    
    # Set parameters of the model and fit
    pipeline.set_params(classifier__alpha=0.0005, classifier__max_iter=250)
    pipeline.fit(X_train, y_train)
    
    # Save the model locally
    model_filename = 'model.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(pipeline, model_file)
        
    # Copy to model to GCS 
    EXPORT_PATH = os.path.join(
        output_dir, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    
    gcs_model_path = '{}/{}'.format(EXPORT_PATH, model_filename)
    subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path])
    print('Saved model in: {}'.format(gcs_model_path))


if __name__ == '__main__':
    fire.Fire(train_evaluate)
