from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


def load_checkpoint(ckpt_dir_or_file, session, var_list=None):
    """Load checkpoint.

    Note:
        This function add some useless ops to the graph. It is better
        to use tf.train.init_from_checkpoint(...).
    """
    if os.path.isdir(ckpt_dir_or_file):
        ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    restorer = tf.train.Saver(var_list)
    restorer.restore(session, ckpt_dir_or_file)
    print(' [*] Loading checkpoint succeeds! Copy variables from % s!' % ckpt_dir_or_file)


def init_from_checkpoint(ckpt_dir_or_file, assignment_map={'/': '/'}):
    tf.train.init_from_checkpoint(ckpt_dir_or_file, assignment_map)
    print(' [*] Loading checkpoint succeeds! Copy variables from % s!' % ckpt_dir_or_file)
