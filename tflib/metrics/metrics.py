import tensorflow as tf


def resettable_metric(metric_fn, metric_params, scope=None):
    with tf.variable_scope(scope, 'resettable_metric') as sc:
        metric_returns = metric_fn(**metric_params)
        reset_op = tf.variables_initializer(tf.local_variables(sc.name))
    return metric_returns + (reset_op,)


def make_resettable(metric_fn, scope=None):
    def resettable_metric_fn(*args, **kwargs):
        with tf.variable_scope(scope, 'resettable_metric') as sc:
            metric_returns = metric_fn(*args, **kwargs)
            reset_op = tf.variables_initializer(tf.local_variables(sc.name))
        return metric_returns + (reset_op,)
    return resettable_metric_fn
