from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
import json
import traceback


import imlib as im
import numpy as np
import pylib
import tensorflow as tf
import tflib as tl

import data
import models


# ==============================================================================
# =                                    param                                   =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', dest='experiment_name', help='experiment_name')
parser.add_argument('--test_atts', dest='test_atts', nargs='+', help='test_atts')
parser.add_argument('--test_ints', dest='test_ints', type=float, nargs='+', help='test_ints')
args_ = parser.parse_args()
with open('./output/%s/setting.txt' % args_.experiment_name) as f:
    args = json.load(f)

# model
atts = args['atts']
n_att = len(atts)
img_size = args['img_size']
shortcut_layers = args['shortcut_layers']
inject_layers = args['inject_layers']
enc_dim = args['enc_dim']
dec_dim = args['dec_dim']
dis_dim = args['dis_dim']
dis_fc_dim = args['dis_fc_dim']
enc_layers = args['enc_layers']
dec_layers = args['dec_layers']
dis_layers = args['dis_layers']
# testing
test_atts = args_.test_atts
thres_int = args['thres_int']
test_ints = args_.test_ints
# others
use_cropped_img = args['use_cropped_img']
experiment_name = args_.experiment_name

assert test_atts is not None, 'test_atts should be chosen in %s' % (str(atts))
for a in test_atts:
    assert a in atts, 'test_atts should be chosen in %s' % (str(atts))

assert len(test_ints) == len(test_atts), 'the lengths of test_ints and test_atts should be the same!'


# ==============================================================================
# =                                   graphs                                   =
# ==============================================================================

# data
sess = tl.session()
te_data = data.Celeba('./data', atts, img_size, 1, part='test', sess=sess, crop=not use_cropped_img)

# models
Genc = partial(models.Genc, dim=enc_dim, n_layers=enc_layers)
Gdec = partial(models.Gdec, dim=dec_dim, n_layers=dec_layers, shortcut_layers=shortcut_layers, inject_layers=inject_layers)

# inputs
xa_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])

# sample
x_sample = Gdec(Genc(xa_sample, is_training=False), _b_sample, is_training=False)


# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# initialization
ckpt_dir = './output/%s/checkpoints' % experiment_name
try:
    tl.load_checkpoint(ckpt_dir, sess)
except:
    raise Exception(' [*] No checkpoint!')

# sample
try:
    for idx, batch in enumerate(te_data):
        xa_sample_ipt = batch[0]
        a_sample_ipt = batch[1]
        b_sample_ipt = np.array(a_sample_ipt, copy=True)
        for a in test_atts:
            i = atts.index(a)
            b_sample_ipt[:, i] = 1 - b_sample_ipt[:, i]   # inverse attribute
            b_sample_ipt = data.Celeba.check_attribute_conflict(b_sample_ipt, atts[i], atts)

        x_sample_opt_list = [xa_sample_ipt, np.full((1, img_size, img_size // 10, 3), -1.0)]
        _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
        for a, i in zip(test_atts, test_ints):
            _b_sample_ipt[..., atts.index(a)] = _b_sample_ipt[..., atts.index(a)] * i / thres_int
        x_sample_opt_list.append(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt, _b_sample: _b_sample_ipt}))
        sample = np.concatenate(x_sample_opt_list, 2)

        save_dir = './output/%s/sample_testing_multi_%s' % (experiment_name, str(test_atts))
        pylib.mkdir(save_dir)
        im.imwrite(sample.squeeze(0), '%s/%d.png' % (save_dir, idx + 182638))

        print('%d.png done!' % (idx + 182638))

except:
    traceback.print_exc()
finally:
    sess.close()
