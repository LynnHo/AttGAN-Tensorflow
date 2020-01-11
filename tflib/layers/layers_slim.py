# functions compatible with tensorflow.contrib

import six

import tensorflow as tf

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope


@add_arg_scope
def fully_connected(inputs,
                    num_outputs,
                    activation_fn=nn.relu,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_normalizer_fn=None,
                    weights_normalizer_params=None,
                    weights_initializer=initializers.xavier_initializer(),
                    weights_regularizer=None,
                    biases_initializer=init_ops.zeros_initializer(),
                    biases_regularizer=None,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    scope=None):
    # Be copied and modified from tensorflow-0.12.0.contrib.layer.fully_connected,
    # add weights_nomalizer_* options.
    """Adds a fully connected layer.

    `fully_connected` creates a variable called `weights`, representing a fully
    connected weight matrix, which is multiplied by the `inputs` to produce a
    `Tensor` of hidden units. If a `normalizer_fn` is provided (such as
    `batch_norm`), it is then applied. Otherwise, if `normalizer_fn` is
    None and a `biases_initializer` is provided then a `biases` variable would be
    created and added the hidden units. Finally, if `activation_fn` is not `None`,
    it is applied to the hidden units as well.

    Note: that if `inputs` have a rank greater than 2, then `inputs` is flattened
    prior to the initial matrix multiply by `weights`.

    Args:
      inputs: A tensor of with at least rank 2 and value for the last dimension,
        i.e. `[batch_size, depth]`, `[None, None, None, channels]`.
      num_outputs: Integer or long, the number of output units in the layer.
      activation_fn: activation function, set to None to skip it and maintain
        a linear activation.
      normalizer_fn: normalization function to use instead of `biases`. If
        `normalizer_fn` is provided then `biases_initializer` and
        `biases_regularizer` are ignored and `biases` are not created nor added.
        default set to None for no normalizer function
      normalizer_params: normalization function parameters.
      weights_normalizer_fn: weights normalization function.
      weights_normalizer_params: weights normalization function parameters.
      weights_initializer: An initializer for the weights.
      weights_regularizer: Optional regularizer for the weights.
      biases_initializer: An initializer for the biases. If None skip biases.
      biases_regularizer: Optional regularizer for the biases.
      reuse: whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: Optional list of collections for all the variables or
        a dictionary containing a different list of collections per variable.
      outputs_collections: collection to add the outputs.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      scope: Optional scope for variable_scope.

    Returns:
       the tensor variable representing the result of the series of operations.

    Raises:
      ValueError: if x has rank less than 2 or if its last dimension is not set.
    """
    if not (isinstance(num_outputs, six.integer_types)):
        raise ValueError('num_outputs should be int or long, got %s.', num_outputs)
    with variable_scope.variable_scope(scope, 'fully_connected', [inputs],
                                       reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        dtype = inputs.dtype.base_dtype
        inputs_shape = inputs.get_shape()
        num_input_units = utils.last_dimension(inputs_shape, min_rank=2)

        static_shape = inputs_shape.as_list()
        static_shape[-1] = num_outputs

        out_shape = array_ops.unpack(array_ops.shape(inputs), len(static_shape))
        out_shape[-1] = num_outputs

        weights_shape = [num_input_units, num_outputs]
        weights_collections = utils.get_variable_collections(
            variables_collections, 'weights')
        weights = variables.model_variable('weights',
                                           shape=weights_shape,
                                           dtype=dtype,
                                           initializer=weights_initializer,
                                           regularizer=weights_regularizer,
                                           collections=weights_collections,
                                           trainable=trainable)
        if weights_normalizer_fn is not None:
            weights_normalizer_params = weights_normalizer_params or {}
            weights = weights_normalizer_fn(weights, **weights_normalizer_params)
        if len(static_shape) > 2:
            # Reshape inputs
            inputs = array_ops.reshape(inputs, [-1, num_input_units])
        outputs = standard_ops.matmul(inputs, weights)
        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)
        else:
            if biases_initializer is not None:
                biases_collections = utils.get_variable_collections(
                    variables_collections, 'biases')
                biases = variables.model_variable('biases',
                                                  shape=[num_outputs, ],
                                                  dtype=dtype,
                                                  initializer=biases_initializer,
                                                  regularizer=biases_regularizer,
                                                  collections=biases_collections,
                                                  trainable=trainable)
                outputs = nn.bias_add(outputs, biases)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        if len(static_shape) > 2:
            # Reshape back outputs
            outputs = array_ops.reshape(outputs, array_ops.pack(out_shape))
            outputs.set_shape(static_shape)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)


@add_arg_scope
def convolution(inputs,
                num_outputs,
                kernel_size,
                stride=1,
                padding='SAME',
                data_format=None,
                rate=1,
                activation_fn=nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_normalizer_fn=None,
                weights_normalizer_params=None,
                weights_initializer=initializers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=init_ops.zeros_initializer(),
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None):
    # Be copied and modified from tensorflow-0.12.0.contrib.layer.convolution,
    # add weights_nomalizer_* options.
    """Adds an N-D convolution followed by an optional batch_norm layer.

    It is required that 1 <= N <= 3.

    `convolution` creates a variable called `weights`, representing the
    convolutional kernel, that is convolved (actually cross-correlated) with the
    `inputs` to produce a `Tensor` of activations. If a `normalizer_fn` is
    provided (such as `batch_norm`), it is then applied. Otherwise, if
    `normalizer_fn` is None and a `biases_initializer` is provided then a `biases`
    variable would be created and added the activations. Finally, if
    `activation_fn` is not `None`, it is applied to the activations as well.

    Performs a'trous convolution with input stride/dilation rate equal to `rate`
    if a value > 1 for any dimension of `rate` is specified.  In this case
    `stride` values != 1 are not supported.

    Args:
      inputs: a Tensor of rank N+2 of shape
        `[batch_size] + input_spatial_shape + [in_channels]` if data_format does
        not start with "NC" (default), or
        `[batch_size, in_channels] + input_spatial_shape` if data_format starts
        with "NC".
      num_outputs: integer, the number of output filters.
      kernel_size: a sequence of N positive integers specifying the spatial
        dimensions of of the filters.  Can be a single integer to specify the same
        value for all spatial dimensions.
      stride: a sequence of N positive integers specifying the stride at which to
        compute output.  Can be a single integer to specify the same value for all
        spatial dimensions.  Specifying any `stride` value != 1 is incompatible
        with specifying any `rate` value != 1.
      padding: one of `"VALID"` or `"SAME"`.
      data_format: A string or None.  Specifies whether the channel dimension of
        the `input` and output is the last dimension (default, or if `data_format`
        does not start with "NC"), or the second dimension (if `data_format`
        starts with "NC").  For N=1, the valid values are "NWC" (default) and
        "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".  For
        N=3, currently the only valid value is "NDHWC".
      rate: a sequence of N positive integers specifying the dilation rate to use
        for a'trous convolution.  Can be a single integer to specify the same
        value for all spatial dimensions.  Specifying any `rate` value != 1 is
        incompatible with specifying any `stride` value != 1.
      activation_fn: activation function, set to None to skip it and maintain
        a linear activation.
      normalizer_fn: normalization function to use instead of `biases`. If
        `normalizer_fn` is provided then `biases_initializer` and
        `biases_regularizer` are ignored and `biases` are not created nor added.
        default set to None for no normalizer function
      normalizer_params: normalization function parameters.
      weights_normalizer_fn: weights normalization function.
      weights_normalizer_params: weights normalization function parameters.
      weights_initializer: An initializer for the weights.
      weights_regularizer: Optional regularizer for the weights.
      biases_initializer: An initializer for the biases. If None skip biases.
      biases_regularizer: Optional regularizer for the biases.
      reuse: whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: optional list of collections for all the variables or
        a dictionary containing a different list of collection per variable.
      outputs_collections: collection to add the outputs.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      scope: Optional scope for `variable_scope`.

    Returns:
      a tensor representing the output of the operation.

    Raises:
      ValueError: if `data_format` is invalid.
      ValueError: both 'rate' and `stride` are not uniformly 1.
    """
    if data_format not in [None, 'NWC', 'NCW', 'NHWC', 'NCHW', 'NDHWC']:
        raise ValueError('Invalid data_format: %r' % (data_format,))
    with variable_scope.variable_scope(scope, 'Conv', [inputs],
                                       reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        dtype = inputs.dtype.base_dtype
        input_rank = inputs.get_shape().ndims
        if input_rank is None:
            raise ValueError('Rank of inputs must be known')
        if input_rank < 3 or input_rank > 5:
            raise ValueError('Rank of inputs is %d, which is not >= 3 and <= 5' %
                             input_rank)
        conv_dims = input_rank - 2
        kernel_size = utils.n_positive_integers(conv_dims, kernel_size)
        stride = utils.n_positive_integers(conv_dims, stride)
        rate = utils.n_positive_integers(conv_dims, rate)

        if data_format is None or data_format.endswith('C'):
            num_input_channels = inputs.get_shape()[input_rank - 1].value
        elif data_format.startswith('NC'):
            num_input_channels = inputs.get_shape()[1].value
        else:
            raise ValueError('Invalid data_format')

        if num_input_channels is None:
            raise ValueError('Number of in_channels must be known.')

        weights_shape = (
            list(kernel_size) + [num_input_channels, num_outputs])
        weights_collections = utils.get_variable_collections(variables_collections,
                                                             'weights')
        weights = variables.model_variable('weights',
                                           shape=weights_shape,
                                           dtype=dtype,
                                           initializer=weights_initializer,
                                           regularizer=weights_regularizer,
                                           collections=weights_collections,
                                           trainable=trainable)
        if weights_normalizer_fn is not None:
            weights_normalizer_params = weights_normalizer_params or {}
            weights = weights_normalizer_fn(weights, **weights_normalizer_params)
        outputs = nn.convolution(input=inputs,
                                 filter=weights,
                                 dilation_rate=rate,
                                 strides=stride,
                                 padding=padding,
                                 data_format=data_format)
        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)
        else:
            if biases_initializer is not None:
                biases_collections = utils.get_variable_collections(
                    variables_collections, 'biases')
                biases = variables.model_variable('biases',
                                                  shape=[num_outputs],
                                                  dtype=dtype,
                                                  initializer=biases_initializer,
                                                  regularizer=biases_regularizer,
                                                  collections=biases_collections,
                                                  trainable=trainable)
                outputs = nn.bias_add(outputs, biases, data_format=data_format)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)


convolution2d = convolution
convolution3d = convolution


@add_arg_scope
def spectral_normalization(weights,
                           num_iterations=1,
                           epsilon=1e-12,
                           u_initializer=tf.random_normal_initializer(),
                           updates_collections=tf.GraphKeys.UPDATE_OPS,
                           is_training=True,
                           reuse=None,
                           variables_collections=None,
                           outputs_collections=None,
                           scope=None):
    with tf.variable_scope(scope, 'SpectralNorm', [weights], reuse=reuse) as sc:
        weights = tf.convert_to_tensor(weights)

        dtype = weights.dtype.base_dtype

        w_t = tf.reshape(weights, [-1, weights.shape.as_list()[-1]])
        w = tf.transpose(w_t)
        m, n = w.shape.as_list()

        u_collections = utils.get_variable_collections(variables_collections, 'u')
        u = tf.get_variable("u",
                            shape=[m, 1],
                            dtype=dtype,
                            initializer=u_initializer,
                            trainable=False,
                            collections=u_collections,)
        sigma_collections = utils.get_variable_collections(variables_collections, 'sigma')
        sigma = tf.get_variable('sigma',
                                shape=[],
                                dtype=dtype,
                                initializer=tf.zeros_initializer(),
                                trainable=False,
                                collections=sigma_collections)

        def _power_iteration(i, u, v):
            v_ = tf.nn.l2_normalize(tf.matmul(w_t, u), epsilon=epsilon)
            u_ = tf.nn.l2_normalize(tf.matmul(w, v_), epsilon=epsilon)
            return i + 1, u_, v_

        _, u_, v_ = tf.while_loop(
            cond=lambda i, _1, _2: i < num_iterations,
            body=_power_iteration,
            loop_vars=[tf.constant(0), u, tf.zeros(shape=[n, 1], dtype=tf.float32)]
        )
        u_ = tf.stop_gradient(u_)
        v_ = tf.stop_gradient(v_)
        sigma_ = tf.matmul(tf.transpose(u_), tf.matmul(w, v_))[0, 0]

        update_u = u.assign(u_)
        update_sigma = sigma.assign(sigma_)
        if updates_collections is None:
            def _force_update():
                with tf.control_dependencies([update_u, update_sigma]):
                    return tf.identity(sigma_)

            sigma_ = utils.smart_cond(is_training, _force_update, lambda: sigma)
            weights_sn = weights / sigma_
        else:
            sigma_ = utils.smart_cond(is_training, lambda: sigma_, lambda: sigma)
            weights_sn = weights / sigma_
            tf.add_to_collections(updates_collections, update_u)
            tf.add_to_collections(updates_collections, update_sigma)

        return utils.collect_named_outputs(outputs_collections, sc.name, weights_sn)


# Simple alias.
conv2d = convolution2d
conv3d = convolution3d
