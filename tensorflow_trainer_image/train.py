
"""Census Tensorflow classifier trainer script."""

import pickle
import subprocess
import sys
import fire
import pandas as pd
import tensorflow as tf
import datetime
import os

CSV_COLUMNS = ["age",
               "workclass",
               "education_num",
               "occupation",
               "hours_per_week",
               "income_bracket"]

# Add string name for label column
LABEL_COLUMN = "income_bracket"

# Set default values for each CSV column as a list of lists.
# Treat is_male and plurality as strings.
DEFAULTS = [[18], ["?"], [4], ["?"], [20],["<=50K"]]

def features_and_labels(row_data):
    cols = tf.io.decode_csv(row_data, record_defaults=DEFAULTS)
    feats = {
        'age': tf.reshape(cols[0], [1,]),
        'workclass': tf.reshape(cols[1],[1,]),
        'education_num': tf.reshape(cols[2],[1,]),
        'occupation': tf.reshape(cols[3],[1,]),
        'hours_per_week': tf.reshape(cols[4],[1,]),
        'income_bracket': cols[5]
    }
    label = feats.pop('income_bracket')
    label_int = tf.case([(tf.math.equal(label,tf.constant([' <=50K'])), lambda: 0),
                        (tf.math.equal(label,tf.constant([' >50K'])), lambda: 1)])
    
    return feats, label_int

def load_dataset(pattern, batch_size=1, mode='eval'):
    # Make a CSV dataset
    filelist = tf.io.gfile.glob(pattern)
    dataset = tf.data.TextLineDataset(filelist).skip(1)
    dataset = dataset.map(features_and_labels)

    # Shuffle and repeat for training
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=10*batch_size).batch(batch_size).repeat()
    else:
        dataset = dataset.batch(10)

    return dataset

def train_evaluate(training_dataset_path, validation_dataset_path, batch_size, num_train_examples, num_evals, output_dir):
    inputs = {
        'age': tf.keras.layers.Input(name='age',shape=[None],dtype='int32'),
        'workclass': tf.keras.layers.Input(name='workclass',shape=[None],dtype='string'),
        'education_num': tf.keras.layers.Input(name='education_num',shape=[None],dtype='int32'),
        'occupation': tf.keras.layers.Input(name='occupation',shape=[None],dtype='string'),
        'hours_per_week': tf.keras.layers.Input(name='hours_per_week',shape=[None],dtype='int32')
    }
    
    batch_size = int(batch_size)
    num_train_examples = int(num_train_examples)
    num_evals = int(num_evals)
    
    feat_cols = {
        'age': tf.feature_column.numeric_column('age'),
        'workclass': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key='workclass', hash_bucket_size=100
            )
        ),
        'education_num': tf.feature_column.numeric_column('education_num'),
        'occupation': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key='occupation', hash_bucket_size=100
            )
        ),
        'hours_per_week': tf.feature_column.numeric_column('hours_per_week')
    }
    
    dnn_inputs = tf.keras.layers.DenseFeatures(
        feature_columns=feat_cols.values())(inputs)
    h1 = tf.keras.layers.Dense(64, activation='relu')(dnn_inputs)
    h2 = tf.keras.layers.Dense(128, activation='relu')(h1)
    h3 = tf.keras.layers.Dense(64, activation='relu')(h2)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(h3)
    
    model = tf.keras.models.Model(inputs=inputs,outputs=output)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    
    trainds = load_dataset(
        pattern=training_dataset_path,
        batch_size=batch_size,
        mode='train')
    
    evalds = load_dataset(
        pattern=validation_dataset_path,
        mode='eval')
    
    
    steps_per_epoch = num_train_examples // (batch_size * num_evals)
    
    history = model.fit(
        trainds,
        validation_data=evalds,
        validation_steps=100,
        epochs=num_evals,
        steps_per_epoch=steps_per_epoch
    )
    
    EXPORT_PATH = os.path.join(
    output_dir, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    tf.saved_model.save(
        obj=model, export_dir=EXPORT_PATH)  # with default serving function
    
    print("Exported trained model to {}".format(EXPORT_PATH))
    
if __name__ == '__main__':
    fire.Fire(train_evaluate)
