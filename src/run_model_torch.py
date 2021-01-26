import os

import data_loader
import model_runner

predictor_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11']
label_name = 'hm'
# DRY this
feature_names = predictor_names + [label_name]

data_dir = './data'
training_data_path = os.path.join(data_dir, 'sample_data_??.tfrecord.gz')

train_size = 2500
batch_size = 8

training_data = data_loader.load_data_pytorch(
        path=training_data_path,
        predictors=predictor_names,
        label=label_name
        )

print(iter(training_data).take(1).next())
