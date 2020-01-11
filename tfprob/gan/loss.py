import tensorflow as tf


def get_gan_losses_fn():
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def d_loss_fn(r_logit, f_logit):
        r_loss = bce(tf.ones_like(r_logit), r_logit)
        f_loss = bce(tf.zeros_like(f_logit), f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = bce(tf.ones_like(f_logit), f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_v1_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = tf.reduce_mean(tf.maximum(1 - r_logit, 0))
        f_loss = tf.reduce_mean(tf.maximum(1 + f_logit, 0))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = tf.reduce_mean(tf.maximum(1 - f_logit, 0))
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_v2_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = tf.reduce_mean(tf.maximum(1 - r_logit, 0))
        f_loss = tf.reduce_mean(tf.maximum(1 + f_logit, 0))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = tf.reduce_mean(- f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_lsgan_losses_fn():
    mse = tf.keras.losses.MeanSquaredError()

    def d_loss_fn(r_logit, f_logit):
        r_loss = mse(tf.ones_like(r_logit), r_logit)
        f_loss = mse(tf.zeros_like(f_logit), f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = mse(tf.ones_like(f_logit), f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_wgan_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = - tf.reduce_mean(r_logit)
        f_loss = tf.reduce_mean(f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = - tf.reduce_mean(f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_adversarial_losses_fn(mode):
    if mode == 'gan':
        return get_gan_losses_fn()
    elif mode == 'hinge_v1':
        return get_hinge_v1_losses_fn()
    elif mode == 'hinge_v2':
        return get_hinge_v2_losses_fn()
    elif mode == 'lsgan':
        return get_lsgan_losses_fn()
    elif mode == 'wgan':
        return get_wgan_losses_fn()
