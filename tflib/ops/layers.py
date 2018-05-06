from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def flatten_fully_connected(inputs,
                            num_outputs,
                            activation_fn=tf.nn.relu,
                            normalizer_fn=None,
                            normalizer_params=None,
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=None,
                            biases_initializer=tf.zeros_initializer(),
                            biases_regularizer=None,
                            reuse=None,
                            variables_collections=None,
                            outputs_collections=None,
                            trainable=True,
                            scope=None):
    with tf.variable_scope(scope, 'flatten_fully_connected', [inputs]):
        if inputs.shape.ndims > 2:
            inputs = slim.flatten(inputs)
        return slim.fully_connected(inputs,
                                    num_outputs,
                                    activation_fn,
                                    normalizer_fn,
                                    normalizer_params,
                                    weights_initializer,
                                    weights_regularizer,
                                    biases_initializer,
                                    biases_regularizer,
                                    reuse,
                                    variables_collections,
                                    outputs_collections,
                                    trainable,
                                    scope)

flatten_dense = flatten_fully_connected
