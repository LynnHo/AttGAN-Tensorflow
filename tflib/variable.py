from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def tensors_filter(tensors, filters, combine_type='or'):
    assert isinstance(tensors, (list, tuple)), '`tensors` shoule be a list or tuple!'
    assert isinstance(filters, (str, list, tuple)), '`filters` should be a string or a list(tuple) of strings!'
    assert combine_type == 'or' or combine_type == 'and', "`combine_type` should be 'or' or 'and'!"

    if isinstance(filters, str):
        filters = [filters]

    f_tens = []
    for ten in tensors:
        if combine_type == 'or':
            for filt in filters:
                if filt in ten.name:
                    f_tens.append(ten)
                    break
        elif combine_type == 'and':
            all_pass = True
            for filt in filters:
                if filt not in ten.name:
                    all_pass = False
                    break
            if all_pass:
                f_tens.append(ten)
    return f_tens


def global_variables(filters=None, combine_type='or'):
    global_vars = tf.global_variables()
    if filters is None:
        return global_vars
    else:
        return tensors_filter(global_vars, filters, combine_type)


def trainable_variables(filters=None, combine_type='or'):
    t_var = tf.trainable_variables()
    if filters is None:
        return t_var
    else:
        return tensors_filter(t_var, filters, combine_type)
