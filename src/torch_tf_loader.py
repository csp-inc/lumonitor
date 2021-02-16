import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader

def parse_tfrecord(example_proto, features_dict):
    return tf.io.parse_single_example(example_proto, features_dict)

def to_tuple(inputs, features, predictors, label):
    inputs_list = [inputs.get(key) for key in features]
    inputs_stack = tf.stack(inputs_list, axis=0)
    transposed_stack = tf.transpose(inputs_stack, [1, 2, 0])
    return (transposed_stack[:, :, :len(predictors)],
            transposed_stack[:, :, features.index(label)])

def get_features_dict(predictors, label):
    kernel_shape = [256, 256]
    feature = tf.io.FixedLenFeature(
        shape=kernel_shape,
        dtype=tf.float32,
        # There are some missing records, for now just fill w/ 0
        default_value=tf.fill(kernel_shape, 0.0))
    features = predictors + [label]
    columns = [feature for n in features]
    return dict(zip(features, columns))

def load_data(path, predictors, label, train_size=None, batch_size=None):
    features_dict = get_features_dict(predictors, label)

    glob = tf.io.gfile.glob(path)
    d = (tf.data.TFRecordDataset(glob, compression_type='GZIP')
         .map(lambda x: parse_tfrecord(x, features_dict))
         .map(lambda x: to_tuple(x, features, predictors, label)))

    if train_size is not None:
        d = d.shuffle(train_size)

    return d.batch(batch_size).repeat()

def tf_to_torch(predictor_tensor, label_tensor):
    return (torch.Tensor(predictor_tensor.numpy()),
        torch.Tensor(label_tensor.numpy()))

def load_data_pytorch(path, predictors, label):
    features = predictors + [label]
    features_dict = get_features_dict(predictors, label)
    glob = tf.io.gfile.glob(path)

    d = (tf.data.TFRecordDataset(glob, compression_type='GZIP')
         .map(lambda x: parse_tfrecord(x, features_dict))
         .map(lambda x: to_tuple(x, features, predictors, label))
         .map(tf_to_torch))

    return DataLoader(TensorDataSet(d))
