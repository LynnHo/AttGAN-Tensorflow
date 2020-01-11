import tensorflow as tf


def gaussian_kernel2d(kernel_radias, std):
    d = tf.distributions.Normal(0.0, float(std))
    vals = d.prob(tf.range(start=-kernel_radias, limit=kernel_radias + 1, dtype=tf.float32))
    kernel = vals[:, None] * vals[None, :]
    kernel /= tf.reduce_sum(kernel)
    return kernel


def filter2d(image, kernel, padding, data_format=None):
    kernel = kernel[:, :, None, None]
    if data_format is None or data_format == "NHWC":
        kernel = tf.tile(kernel, [1, 1, image.shape[3], 1])
    elif data_format == "NCHW":
        kernel = tf.tile(kernel, [1, 1, image.shape[1], 1])
    return tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding=padding, data_format=data_format)


def gaussian_filter2d(image, kernel_radias, std, padding, data_format=None):
    kernel = gaussian_kernel2d(kernel_radias, std)
    return filter2d(image, kernel, padding, data_format=None)
