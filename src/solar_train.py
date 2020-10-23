import tensorflow as tf
from tensorflow import keras
import solar_constants as sc

train_dataset = tf.data.TFRecordDataset('gs://aft-saf/solar_training.tfrecord.gz', compression_type='GZIP')

columns = [
  tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for k in sc.FEATURE_NAMES
]

# Dictionary with names as keys, features as values.
features_dict = dict(zip(sc.FEATURE_NAMES, columns))


def parse_tfrecord(example_proto):
    """The parsing function.

    Read a serialized example into the structure defined by featuresDict.

    Args:
    example_proto: a serialized Example.

    Returns:
    A tuple of the predictors dictionary and the label, cast to an `int32`.
    """
    print(example_proto)
    parsed_features = tf.io.parse_single_example(example_proto, features_dict)
    labels = parsed_features.pop('solar')
    return parsed_features, tf.cast(labels, tf.int32)

# Map the function over the dataset.
input_dataset = train_dataset.map(parse_tfrecord, num_parallel_calls=5)

# Keras requires inputs as a tuple.  Note that the inputs must be in the
# right shape.  Also note that to use the categorical_crossentropy loss,
# the label needs to be turned into a one-hot vector.
def to_tuple(inputs, label):
  return (tf.transpose(list(inputs.values())),
          tf.one_hot(indices=label, depth=sc.N_CLASSES))

# Map the to_tuple function, shuffle and batch.
input_dataset = input_dataset.map(to_tuple).batch(8)

# Define the layers in the model.
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(sc.N_CLASSES, activation=tf.nn.softmax)
])

# Compile the model with the specified loss function.
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model to the training data.
model.fit(x=input_dataset, epochs=50)

test_dataset = (
  tf.data.TFRecordDataset('gs://aft-saf/solar_test.tfrecord.gz', compression_type='GZIP')
    .map(parse_tfrecord, num_parallel_calls=5)
    .map(to_tuple)
    .batch(1))

# It sucks, but whatevs
model.evaluate(test_dataset)

MODEL_DIR = 'gs://aft-saf/demo_pixel_model'
model.save(MODEL_DIR, save_format='tf')

from tensorflow.python.tools import saved_model_utils

meta_graph_def = saved_model_utils.get_meta_graph_def(MODEL_DIR, 'serve')
inputs = meta_graph_def.signature_def['serving_default'].inputs
outputs = meta_graph_def.signature_def['serving_default'].outputs

# Just get the first thing(s) from the serving signature def.  i.e. this
# model only has a single input and a single output.
input_name = None
for k,v in inputs.items():
  input_name = v.name
  break

output_name = None
for k,v in outputs.items():
  output_name = v.name
  break

# Make a dictionary that maps Earth Engine outputs and inputs to
# AI Platform inputs and outputs, respectively.
import json
input_dict = "'" + json.dumps({input_name: "array"}) + "'"
output_dict = "'" + json.dumps({output_name: "solar"}) + "'"
print(input_dict)
print(output_dict)

#earthengine model prepare --source_dir gs://aft-saf/demo_pixel_model --dest_dir gs://aft-saf/eeified_pixel_model --input '{"serving_default_dense_input:0": "array"}' --output '{"StatefulPartitionedCall:0": "solar"}'

# gcloud ai-platform models create 'solar_test' --project 'csp-projects'

# gcloud ai-platform versions create 'v002' --project 'csp-projects' --model 'solar_test' --origin 'gs://aft-saf/eeified_pixel_model' --framework "TENSORFLOW" --runtime-version=2.2 --python-version=3.7
