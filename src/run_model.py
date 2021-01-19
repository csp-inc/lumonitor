import os

import tensorflow as tf

import data_loader
import model_runner

predictor_names = ['B1','B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11']
label_name = 'hm'
# DRY this
feature_names = predictor_names + [ label_name ]

data_dir = './data'
training_data_path = os.path.join(data_dir, 'sample_data_??.tfrecord.gz')

train_size = 2500
batch_size = 8

training_data = data_loader.load_data(
        path=training_data_path, 
        predictors=predictor_names,
        label=label_name,
        train_size=train_size,
        batch_size=batch_size
        )

testing_data_path = os.path.join(data_dir, 'sample_data_?.tfrecord.gz')
testing_data = data_loader.load_data(
        path=testing_data_path,
        predictors=predictor_names,
        label=label_name,
        batch_size=1
        )

optimizer = 'SGD'
loss_function = 'MeanSquaredError'
metrics = ['RootMeanSquaredError']

model = model_runner.get_model(
        predictors=predictor_names,
        optimizer=optimizer,
        loss_function=loss_function,
        metric_names=metrics
        )

epochs = 50
eval_size = 8000

model.fit(
    x=training_data,
    epochs=epochs,
    steps_per_epoch=int(train_size / batch_size),
    validation_data=testing_data,
    validation_steps=eval_size)
