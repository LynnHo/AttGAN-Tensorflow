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
parser.add_argument('--test_att', dest='test_att', help='test_att')
parser.add_argument('--test_int_min', dest='test_int_min', type=float, default=-1.0, help='test_int_min')
parser.add_argument('--test_int_max', dest='test_int_max', type=float, default=1.0, help='test_int_max')
parser.add_argument('--n_slide', dest='n_slide', type=int, default=10, help='n_slide')
args_ = parser.parse_args()
with open('./output/%s/setting.txt' % args_.experiment_name) as f:
    args = json.load(f)

atts = args['atts']
n_att = len(atts)
test_att = args_.test_att
img_size = args['img_size']
thres_int = args['thres_int']
test_int_min = args_.test_int_min
test_int_max = args_.test_int_max
n_slide = args_.n_slide
shortcut_layers = args['shortcut_layers']
inject_layers = args['inject_layers']
experiment_name = args_.experiment_name

assert test_att is not None, 'test_att should be chosen in %s' % (str(atts))


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
        b_sample_ipt = batch[1]

        x_sample_opt_list = [xa_sample_ipt, np.full((1, img_size, img_size // 10, 3), -1.0)]
        for i in range(n_slide):
            test_int = (test_int_max - test_int_min) / (n_slide - 1) * i + test_int_min
            _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
            _b_sample_ipt[..., atts.index(test_att)] = test_int
            x_sample_opt_list.append(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt, _b_sample: _b_sample_ipt}))
        sample = np.concatenate(x_sample_opt_list, 2)

        save_dir = './output/%s/sample_testing_slide_%s' % (experiment_name, test_att)
        pylib.mkdir(save_dir)
        im.imwrite(sample.squeeze(0), '%s/%d.png' % (save_dir, idx + 182638))

        print('%d.png done!' % (idx + 182638))

except:
    traceback.print_exc()
finally:
    sess.close()
