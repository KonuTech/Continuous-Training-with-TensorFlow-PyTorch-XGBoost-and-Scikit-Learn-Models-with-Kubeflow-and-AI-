
import os
import kfp
from kfp.dsl.types import GCPProjectID
from kfp.dsl.types import GCPRegion
from kfp.dsl.types import GCSPath
from kfp.dsl.types import String
from kfp.gcp import use_gcp_secret
import kfp.components as comp
import kfp.dsl as dsl
import kfp.gcp as gcp
import json

# We will use environment vars to set the trainer image names and bucket name
TF_TRAINER_IMAGE = os.getenv('TF_TRAINER_IMAGE')
SCIKIT_TRAINER_IMAGE = os.getenv('SCIKIT_TRAINER_IMAGE')
TORCH_TRAINER_IMAGE = os.getenv('TORCH_TRAINER_IMAGE')
XGB_TRAINER_IMAGE = os.getenv('XGB_TRAINER_IMAGE')
BUCKET = os.getenv('BUCKET')

# Paths to export the training/validation data from bigquery
TRAINING_OUTPUT_PATH = BUCKET + '/census/data/training.csv'
VALIDATION_OUTPUT_PATH = BUCKET + '/census/data/validation.csv'

COMPONENT_URL_SEARCH_PREFIX = 'https://raw.githubusercontent.com/kubeflow/pipelines/0.2.5/components/gcp/'

# Create component factories
component_store = kfp.components.ComponentStore(
    local_search_paths=None, url_search_prefixes=[COMPONENT_URL_SEARCH_PREFIX])

# TODO: Load BigQuery and AI Platform Training ops from component_store
# as bigquery_query_op and mlengine_train_op 

bigquery_query_op = component_store.load_component('bigquery/query')
mlengine_train_op = component_store.load_component('ml_engine/train')

def get_query(dataset='training'):
    """Function that returns either training or validation query"""
    if dataset=='training':
        split = "MOD(ABS(FARM_FINGERPRINT(CAST(functional_weight AS STRING))), 100) < 80"
    elif dataset=='validation':
        split = """MOD(ABS(FARM_FINGERPRINT(CAST(functional_weight AS STRING))), 100) >= 80 
        AND MOD(ABS(FARM_FINGERPRINT(CAST(functional_weight AS STRING))), 100) < 90"""
    else:
        split = "MOD(ABS(FARM_FINGERPRINT(CAST(functional_weight AS STRING))), 100) >= 90"
        
    query = """SELECT age, workclass, education_num, occupation, hours_per_week,income_bracket 
    FROM census.data 
    WHERE {0}""".format(split)
    
    return query

# We will use the training/validation queries as inputs to our pipeline
# This lets us change the training/validation datasets if we wish by simply
# Changing the query. 
TRAIN_QUERY = get_query(dataset='training')
VALIDATION_QUERY=get_query(dataset='validation')

@dsl.pipeline(
    name='Continuous Training with Multiple Frameworks',
    description='Pipeline to create training/validation splits w/ BigQuery then launch multiple AI Platform Training Jobs'
)
def pipeline(
    project_id,
    train_query=TRAIN_QUERY,
    validation_query=VALIDATION_QUERY,
    region='us-central1'
):
    # Creating the training data split
    create_training_split = bigquery_query_op(
        query=train_query,
        project_id=project_id,
        output_gcs_path=TRAINING_OUTPUT_PATH
    ).set_display_name('BQ Train Split')
    
    # TODO: Create the validation data split
    create_validation_split = bigquery_query_op(
        query=validation_query,
        project_id=project_id,
        output_gcs_path=VALIDATION_OUTPUT_PATH
    ).set_display_name('BQ Eval Split')
    
    # These are the output directories where our models will be saved
    tf_output_dir = BUCKET + '/census/models/tf'
    scikit_output_dir = BUCKET + '/census/models/scikit'
    torch_output_dir = BUCKET + '/census/models/torch'
    xgb_output_dir = BUCKET + '/census/models/xgb'
    
    # Training arguments to be passed to the TF Trainer
    tf_args = [
        '--training_dataset_path', create_training_split.outputs['output_gcs_path'],
        '--validation_dataset_path', create_validation_split.outputs['output_gcs_path'],
        '--output_dir', tf_output_dir,
        '--batch_size', '32', 
        '--num_train_examples', '1000',
        '--num_evals', '10'
    ]
    
    # TODO: Fill in the list of the training arguments to be passed to the Scikit-learn Trainer
    scikit_args = [
        '--training_dataset_path', create_training_split.outputs['output_gcs_path'],
        '--validation_dataset_path', create_validation_split.outputs['output_gcs_path'],
        '--output_dir', scikit_output_dir
    ]
    
    # Training arguments to be passed to the PyTorch Trainer
    torch_args = [
        '--training_dataset_path', create_training_split.outputs['output_gcs_path'],
        '--validation_dataset_path', create_validation_split.outputs['output_gcs_path'],
        '--output_dir', torch_output_dir,
        '--batch_size', '32', 
        '--num_epochs', '15',
    ]
    
    # Training arguments to be passed to the XGBoost Trainer 
    xgb_args = [
        '--training_dataset_path', create_training_split.outputs['output_gcs_path'],
        '--validation_dataset_path', create_validation_split.outputs['output_gcs_path'],
        '--output_dir', xgb_output_dir,
        '--max_depth', '10', 
        '--n_estimators', '100'
    ]
    
    # AI Platform Training Jobs with all 4 trainer images 
    
    train_scikit = mlengine_train_op(
        project_id=project_id,
        region=region,
        master_image_uri=SCIKIT_TRAINER_IMAGE,
        args=scikit_args).set_display_name('Scikit-learn Model - AI Platform Training')
    
    train_tf = mlengine_train_op(
        project_id=project_id,
        region=region,
        master_image_uri=TF_TRAINER_IMAGE,
        args=tf_args).set_display_name('Tensorflow Model - AI Platform Training')
    
    train_torch = mlengine_train_op(
        project_id=project_id,
        region=region,
        master_image_uri=TORCH_TRAINER_IMAGE,
        args=torch_args).set_display_name('Pytorch Model - AI Platform Training')
    
    # TODO: Provide arguments to mlengine_train_op to train the XGBoost model
    train_xgb = mlengine_train_op(
        project_id=project_id,
        region=region,
        master_image_uri=XGB_TRAINER_IMAGE,
        args=xgb_args).set_display_name('XGBoost Model - AI Platform Training')
    
