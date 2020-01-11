import tensorflow as tf


def minmax_norm(x, epsilon=1e-12):
    x = tf.to_float(x)
    min_val = tf.reduce_min(x)
    max_val = tf.reduce_max(x)
    norm_x = (x - min_val) / tf.maximum((max_val - min_val), epsilon)
    return norm_x


def reshape(x, shape):
    x = tf.convert_to_tensor(x)
    shape = [x.shape[i] if shape[i] == 0 else shape[i] for i in range(len(shape))]
    shape = [tf.shape(x)[i] if shape[i] is None else shape[i] for i in range(len(shape))]
    return tf.reshape(x, shape)
