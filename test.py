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


""" param """
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', dest='experiment_name', help='experiment_name')
parser.add_argument('--test_int', dest='test_int', type=float, default=1.0, help='test_int')
args_ = parser.parse_args()
with open('./output/%s/setting.txt' % args_.experiment_name) as f:
    args = json.load(f)

atts = args['atts']
n_att = len(atts)
img_size = args['img_size']
thres_int = args['thres_int']
test_int = args_.test_int
shortcut_layers = args['shortcut_layers']
inject_layers = args['inject_layers']
experiment_name = args_.experiment_name


""" graphs """
# data
sess = tl.session()
te_data = data.Celeba('./data', atts, img_size, 1, part='test', sess=sess)

# models
if img_size == 64:
    Genc = models.Genc_64
    Gdec = partial(models.Gdec_64, shortcut_layers=shortcut_layers, inject_layers=inject_layers)
else:
    Genc = models.Genc_128
    Gdec = partial(models.Gdec_128, shortcut_layers=shortcut_layers, inject_layers=inject_layers)

# inputs
xa_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])

# sample
x_sample = Gdec(Genc(xa_sample, is_training=False), _b_sample, is_training=False)


""" train """
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
        b_sample_ipt_list = [a_sample_ipt]  # the first is for reconstruction
        for i in range(len(atts)):
            tmp = np.array(a_sample_ipt, copy=True)
            tmp[:, i] = 1 - tmp[:, i]   # inverse attribute
            tmp = data.Celeba.check_attribute_conflict(tmp, atts[i], atts)
            b_sample_ipt_list.append(tmp)

        x_sample_opt_list = [xa_sample_ipt, np.full((1, img_size, img_size // 10, 3), -1.0)]
        for i, b_sample_ipt in enumerate(b_sample_ipt_list):
            _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
            if i > 0:   # i == 0 is for reconstruction
                _b_sample_ipt[..., i - 1] = _b_sample_ipt[..., i - 1] * test_int / thres_int
            x_sample_opt_list.append(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt, _b_sample: _b_sample_ipt}))
        sample = np.concatenate(x_sample_opt_list, 2)

        save_dir = './output/%s/sample_testing' % experiment_name
        pylib.mkdir(save_dir)
        im.imwrite(sample.squeeze(0), '%s/%d.png' % (save_dir, idx + 182638))

        print('%d.png done!' % (idx + 182638))

except:
    traceback.print_exc()
finally:
    sess.close()
