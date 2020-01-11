import tensorflow as tf


def adaptive_instance_normalization(x, gamma, beta, epsilon=1e-5):
    # modified from https://github.com/taki0112/MUNIT-Tensorflow/blob/master/ops.py
    # x: (N, H, W, C), gamma: (N, 1, 1, C), beta: (N, 1, 1, C)

    c_mean, c_var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)

    return gamma * ((x - c_mean) / c_std) + beta
