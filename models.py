from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflib as tl


conv = partial(slim.conv2d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
fc = partial(tl.flatten_fully_connected, activation_fn=None)
relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)
instance_norm = slim.instance_norm


def Genc_128(x, dim=64, is_training=True):
    bn = partial(batch_norm, is_training=is_training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu)

    with tf.variable_scope('Genc', reuse=tf.AUTO_REUSE):
        z0 = conv_bn_lrelu(x, dim * 1, 4, 2)
        z1 = conv_bn_lrelu(z0, dim * 2, 4, 2)
        z2 = conv_bn_lrelu(z1, dim * 4, 4, 2)
        z3 = conv_bn_lrelu(z2, dim * 8, 4, 2)
        z4 = conv_bn_lrelu(z3, dim * 16, 4, 2)
        return [z0, z1, z2, z3, z4]


def Gdec_128(z, _a, dim=64, shortcut_layers=1, inject_layers=0, is_training=True):
    bn = partial(batch_norm, is_training=is_training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu)

    def _concat(z, z_, _a):
        feats = [z]
        if z_ is not None:
            feats.append(z_)
        if _a is not None:
            _a = tf.reshape(_a, [-1, 1, 1, tl.shape(_a)[-1]])
            _a = tf.tile(_a, [1, tl.shape(z)[1], tl.shape(z)[2], 1])
            feats.append(_a)
        return tf.concat(feats, axis=3)

    with tf.variable_scope('Gdec', reuse=tf.AUTO_REUSE):
        z4 = _concat(z[4], None, _a)

        z3 = dconv_bn_relu(z4, dim * 16, 4, 2)
        if shortcut_layers >= 1:
            z3 = _concat(z3, z[3], None)
        if inject_layers >= 1:
            z3 = _concat(z3, None, _a)

        z2 = dconv_bn_relu(z3, dim * 8, 4, 2)
        if shortcut_layers >= 2:
            z2 = _concat(z2, z[2], None)
        if inject_layers >= 2:
            z2 = _concat(z2, None, _a)

        z1 = dconv_bn_relu(z2, dim * 4, 4, 2)
        if shortcut_layers >= 3:
            z1 = _concat(z1, z[1], None)
        if inject_layers >= 3:
            z1 = _concat(z1, None, _a)

        z0 = dconv_bn_relu(z1, dim * 2, 4, 2)
        if shortcut_layers >= 4:
            z0 = _concat(z0, z[0], None)
        if inject_layers >= 4:
            z0 = _concat(z0, None, _a)

        x = tf.nn.tanh(dconv(z0, 3, 4, 2))

        return x


def D_128(x, n_att, dim=64):
    conv_in_lrelu = partial(conv, normalizer_fn=instance_norm, activation_fn=lrelu)

    with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
        y = conv_in_lrelu(x, dim * 1, 4, 2)
        y = conv_in_lrelu(y, dim * 2, 4, 2)
        y = conv_in_lrelu(y, dim * 4, 4, 2)
        y = conv_in_lrelu(y, dim * 8, 4, 2)
        y = conv_in_lrelu(y, dim * 16, 4, 2)

        logit_wgan = lrelu(fc(y, dim * 16))
        logit_wgan = fc(logit_wgan, 1)

        logit_att = lrelu(fc(y, dim * 16))
        logit_att = fc(logit_att, n_att)

        return logit_wgan, logit_att


def Genc_64(x, dim=64, is_training=True):
    bn = partial(batch_norm, is_training=is_training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu)

    with tf.variable_scope('Genc', reuse=tf.AUTO_REUSE):
        z0 = conv_bn_lrelu(x, dim * 1, 5, 2)
        z1 = conv_bn_lrelu(z0, dim * 2, 5, 2)
        z2 = conv_bn_lrelu(z1, dim * 4, 5, 2)
        z3 = conv_bn_lrelu(z2, dim * 8, 5, 2)
        return [z0, z1, z2, z3]


def Gdec_64(z, _a, dim=64, shortcut_layers=1, inject_layers=0, is_training=True):
    bn = partial(batch_norm, is_training=is_training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu)

    def _concat(z, z_, _a):
        feats = [z]
        if z_ is not None:
            feats.append(z_)
        if _a is not None:
            _a = tf.reshape(_a, [-1, 1, 1, tl.shape(_a)[-1]])
            _a = tf.tile(_a, [1, tl.shape(z)[1], tl.shape(z)[2], 1])
            feats.append(_a)
        return tf.concat(feats, axis=3)

    with tf.variable_scope('Gdec', reuse=tf.AUTO_REUSE):
        z3 = _concat(z[3], None, _a)

        z2 = dconv_bn_relu(z3, dim * 8, 5, 2)
        if shortcut_layers >= 1:
            z2 = _concat(z2, z[2], None)
        if inject_layers >= 1:
            z2 = _concat(z2, None, _a)

        z1 = dconv_bn_relu(z2, dim * 4, 5, 2)
        if shortcut_layers >= 2:
            z1 = _concat(z1, z[1], None)
        if inject_layers >= 2:
            z1 = _concat(z1, None, _a)

        z0 = dconv_bn_relu(z1, dim * 2, 5, 2)
        if shortcut_layers >= 3:
            z0 = _concat(z0, z[0], None)
        if inject_layers >= 3:
            z0 = _concat(z0, None, _a)

        z1_ = dconv_bn_relu(z0, dim * 1, 5, 2)

        x = tf.nn.tanh(dconv(z1_, 3, 5, 1))

        return x


def D_64(x, n_att, dim=64):
    conv_in_lrelu = partial(conv, normalizer_fn=instance_norm, activation_fn=lrelu)

    with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
        y = conv_in_lrelu(x, dim * 1, 3, 1)
        y = conv_in_lrelu(y, dim * 1, 5, 2)
        y = conv_in_lrelu(y, dim * 2, 5, 2)
        y = conv_in_lrelu(y, dim * 4, 5, 2)
        y = conv_in_lrelu(y, dim * 8, 5, 2)
        y = conv_in_lrelu(y, dim * 8, 3, 1)

        logit_wgan = lrelu(fc(y, dim * 16))
        logit_wgan = fc(logit_wgan, 1)

        logit_att = lrelu(fc(y, dim * 16))
        logit_att = fc(logit_att, n_att)

        return logit_wgan, logit_att


def gradient_penalty(real, fake, f):
    def interpolate(a, b):
        with tf.name_scope('interpolate'):
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

    with tf.name_scope('gradient_penalty'):
        x = interpolate(real, fake)
        pred = f(x)
        if isinstance(pred, tuple):
            pred = pred[0]

        grad = tf.gradients(pred, x)[0]
        norm = tf.norm(slim.flatten(grad), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)
        return gp
