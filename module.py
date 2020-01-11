import functools

import tensorflow as tf
import tensorflow.contrib.slim as slim

import utils


conv = functools.partial(slim.conv2d, activation_fn=None)
dconv = functools.partial(slim.conv2d_transpose, activation_fn=None)
fc = functools.partial(slim.fully_connected, activation_fn=None)


class UNetGenc:

    def __call__(self, x, dim=64, n_downsamplings=5, weight_decay=0.0,
                 norm_name='batch_norm', training=True, scope='UNetGenc'):
        MAX_DIM = 1024

        conv_ = functools.partial(conv, weights_regularizer=slim.l2_regularizer(weight_decay))
        norm = utils.get_norm_layer(norm_name, training, updates_collections=None)

        conv_norm_lrelu = functools.partial(conv_, normalizer_fn=norm, activation_fn=tf.nn.leaky_relu)

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            z = x
            zs = []
            for i in range(n_downsamplings):
                d = min(dim * 2**i, MAX_DIM)
                z = conv_norm_lrelu(z, d, 4, 2)
                zs.append(z)

        # variables and update operations
        self.variables = tf.global_variables(scope)
        self.trainable_variables = tf.trainable_variables(scope)
        self.reg_losses = tf.losses.get_regularization_losses(scope)

        return zs


class UNetGdec:

    def __call__(self, zs, a, dim=64, n_upsamplings=5, shortcut_layers=1, inject_layers=1, weight_decay=0.0,
                 norm_name='batch_norm', training=True, scope='UNetGdec'):
        MAX_DIM = 1024

        dconv_ = functools.partial(dconv, weights_regularizer=slim.l2_regularizer(weight_decay))
        norm = utils.get_norm_layer(norm_name, training, updates_collections=None)

        dconv_norm_relu = functools.partial(dconv_, normalizer_fn=norm, activation_fn=tf.nn.relu)

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            a = tf.to_float(a)

            z = utils.tile_concat(zs[-1], a)
            for i in range(n_upsamplings - 1):
                d = min(dim * 2**(n_upsamplings - 1 - i), MAX_DIM)
                z = dconv_norm_relu(z, d, 4, 2)
                if shortcut_layers > i:
                    z = utils.tile_concat([z, zs[-2 - i]])
                if inject_layers > i:
                    z = utils.tile_concat(z, a)
            x = tf.nn.tanh(dconv_(z, 3, 4, 2))

        # variables and update operations
        self.variables = tf.global_variables(scope)
        self.trainable_variables = tf.trainable_variables(scope)
        self.reg_losses = tf.losses.get_regularization_losses(scope)

        return x


class ConvD:

    def __call__(self, x, n_atts, dim=64, fc_dim=1024, n_downsamplings=5, weight_decay=0.0,
                 norm_name='instance_norm', training=True, scope='ConvD'):
        MAX_DIM = 1024

        conv_ = functools.partial(conv, weights_regularizer=slim.l2_regularizer(weight_decay))
        fc_ = functools.partial(fc, weights_regularizer=slim.l2_regularizer(weight_decay))
        norm = utils.get_norm_layer(norm_name, training, updates_collections=None)

        conv_norm_lrelu = functools.partial(conv_, normalizer_fn=norm, activation_fn=tf.nn.leaky_relu)

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            z = x
            for i in range(n_downsamplings):
                d = min(dim * 2**i, MAX_DIM)
                z = conv_norm_lrelu(z, d, 4, 2)
            z = slim.flatten(z)

            logit_gan = tf.nn.leaky_relu(fc_(z, fc_dim))
            logit_gan = fc_(logit_gan, 1)

            logit_att = tf.nn.leaky_relu(fc_(z, fc_dim))
            logit_att = fc_(logit_att, n_atts)

        # variables and update operations
        self.variables = tf.global_variables(scope)
        self.trainable_variables = tf.trainable_variables(scope)
        self.reg_losses = tf.losses.get_regularization_losses(scope)

        return logit_gan, logit_att


def get_model(name, n_atts, weight_decay=0.0):
    if name in ['model_128', 'model_256']:
        Genc = functools.partial(UNetGenc(), dim=64, n_downsamplings=5, weight_decay=weight_decay)
        Gdec = functools.partial(UNetGdec(), dim=64, n_upsamplings=5, shortcut_layers=1, inject_layers=1, weight_decay=weight_decay)
        D = functools.partial(ConvD(), n_atts=n_atts, dim=64, fc_dim=1024, n_downsamplings=5, weight_decay=weight_decay)
    elif name == 'model_384':
        Genc = functools.partial(UNetGenc(), dim=48, n_downsamplings=5, weight_decay=weight_decay)
        Gdec = functools.partial(UNetGdec(), dim=48, n_upsamplings=5, shortcut_layers=1, inject_layers=1, weight_decay=weight_decay)
        D = functools.partial(ConvD(), n_atts=n_atts, dim=48, fc_dim=512, n_downsamplings=5, weight_decay=weight_decay)
    return Genc, Gdec, D
