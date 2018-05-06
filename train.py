from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
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
att_default = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']

parser = argparse.ArgumentParser()
parser.add_argument('--atts', dest='atts', default=att_default, choices=data.Celeba.att_dict.keys(), nargs='+', help='attributes to learn')
parser.add_argument('--img_size', dest='img_size', type=int, default=128, help='size of image')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='# of images per batch')
parser.add_argument('--n_sample', dest='n_sample', type=int, default=64, help='# of sample images')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--n_d', dest='n_d', type=int, default=5, help='# of d updates per g update')
parser.add_argument('--b_distribution', dest='b_distribution', default='none', choices=['none', 'uniform', 'truncated_normal'], help='b_distribution')
parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5, help='thres_int')
parser.add_argument('--test_int', dest='test_int', type=float, default=1.0, help='test_int')
parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1, choices=[0, 1, 2, 3], help='shortcut_layers')
parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0, choices=[0, 1, 2, 3], help='inject_layers')
parser.add_argument('--experiment_name', dest='experiment_name', default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"), help='experiment_name')
args = parser.parse_args()

atts = args.atts
n_att = len(atts)
img_size = args.img_size
epoch = args.epoch
batch_size = args.batch_size
n_sample = args.n_sample
lr_base = args.lr
n_d = args.n_d
b_distribution = args.b_distribution
thres_int = args.thres_int
test_int = args.test_int
shortcut_layers = args.shortcut_layers
inject_layers = args.inject_layers
experiment_name = args.experiment_name

pylib.mkdir('./output/%s' % experiment_name)
with open('./output/%s/setting.txt' % experiment_name, 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))


""" graphs """
# data
sess = tl.session()
tr_data = data.Celeba('./data', atts, img_size, batch_size, part='train', sess=sess)
val_data = data.Celeba('./data', atts, img_size, n_sample, part='val', shuffle=False, sess=sess)

# models
if img_size == 64:
    Genc = models.Genc_64
    Gdec = partial(models.Gdec_64, shortcut_layers=shortcut_layers, inject_layers=inject_layers)
    D = partial(models.D_64, n_att=n_att)
else:
    Genc = models.Genc_128
    Gdec = partial(models.Gdec_128, shortcut_layers=shortcut_layers, inject_layers=inject_layers)
    D = partial(models.D_128, n_att=n_att)

# inputs
lr = tf.placeholder(dtype=tf.float32, shape=[])

xa = tr_data.batch_op[0]
a = tr_data.batch_op[1]
b = tf.random_shuffle(a)
_a = (tf.to_float(a) * 2 - 1) * thres_int
if b_distribution == 'none':
    _b = (tf.to_float(b) * 2 - 1) * thres_int
elif b_distribution == 'uniform':
    _b = (tf.to_float(b) * 2 - 1) * tf.random_uniform(tf.shape(b)) * (2 * thres_int)
elif b_distribution == 'truncated_normal':
    _b = (tf.to_float(b) * 2 - 1) * (tf.truncated_normal(tf.shape(b)) + 2) / 4.0 * (2 * thres_int)

xa_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])

# generate
z = Genc(xa)
xb_ = Gdec(z, _b)
with tf.control_dependencies([xb_]):
    xa_ = Gdec(z, _a)

# discriminate
xa_logit_wgan, xa_logit_att = D(xa)
xb__logit_wgan, xb__logit_att = D(xb_)

# x discriminator losses
x_gp = models.gradient_penalty(xa, xb_, D)
x_wd = tf.reduce_mean(xa_logit_wgan) - tf.reduce_mean(xb__logit_wgan)
xa_loss_att = tf.losses.sigmoid_cross_entropy(a, xa_logit_att)
d_loss = -x_wd + x_gp * 10.0 + xa_loss_att

# x generator losses
xa__loss_rec = tf.losses.absolute_difference(xa, xa_)
xb__loss_att = tf.losses.sigmoid_cross_entropy(b, xb__logit_att)
xb__loss_wgan = -tf.reduce_mean(xb__logit_wgan)
g_loss = xb__loss_wgan + xb__loss_att * 10.0 + xa__loss_rec * 100.0

# optim
d_var = tl.trainable_variables('D')
d_step = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss, var_list=d_var)

g_var = tl.trainable_variables('G')
g_step = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(g_loss, var_list=g_var)

# summary
d_summary = tl.summary({
    x_gp: 'x_gp',
    x_wd: 'x_wd',
    xa_loss_att: 'xa_loss_att',
}, scope='D')

lr_summary = tl.summary({lr: 'lr'}, scope='Learning_Rate')

g_summary = tl.summary({
    xb__loss_wgan: 'xb__loss_wgan',
    xb__loss_att: 'xb__loss_att',
    xa__loss_rec: 'xa__loss_rec',
}, scope='G')

d_summary = tf.summary.merge([d_summary, lr_summary])

# sample
x_sample = Gdec(Genc(xa_sample, is_training=False), _b_sample, is_training=False)


""" train """
# iteration counter
it_cnt, update_cnt = tl.counter()

# saver
saver = tf.train.Saver(max_to_keep=1)

# summary writer
summary_writer = tf.summary.FileWriter('./output/%s/summaries' % experiment_name, sess.graph)

# initialization
ckpt_dir = './output/%s/checkpoints' % experiment_name
pylib.mkdir(ckpt_dir)
try:
    tl.load_checkpoint(ckpt_dir, sess)
except:
    sess.run(tf.global_variables_initializer())

# train
try:
    # data for sampling
    xa_sample_ipt, a_sample_ipt = val_data.get_next()
    b_sample_ipt_list = [a_sample_ipt]  # the first is for reconstruction
    for i in range(len(atts)):
        tmp = np.array(a_sample_ipt, copy=True)
        tmp[:, i] = 1 - tmp[:, i]   # inverse attribute
        tmp = data.Celeba.check_attribute_conflict(tmp, atts[i], atts)
        b_sample_ipt_list.append(tmp)

    it_per_epoch = len(tr_data) // (batch_size * (n_d + 1))
    max_it = epoch * it_per_epoch
    for it in range(sess.run(it_cnt), max_it):
        with pylib.Timer(is_output=False) as t:
            sess.run(update_cnt)

            # which epoch
            epoch = it // it_per_epoch
            it_in_epoch = it % it_per_epoch + 1

            # learning rate
            lr_ipt = lr_base / (10 ** (epoch // 100))

            # train D
            for i in range(n_d):
                d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={lr: lr_ipt})
            summary_writer.add_summary(d_summary_opt, it)

            # train G
            g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={lr: lr_ipt})
            summary_writer.add_summary(g_summary_opt, it)

            # display
            if (it + 1) % 1 == 0:
                print("Epoch: (%3d) (%5d/%5d) Time: %s!" % (epoch, it_in_epoch, it_per_epoch, t))

            # save
            if (it + 1) % 1000 == 0:
                save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_in_epoch, it_per_epoch))
                print('Model is saved at %s!' % save_path)

            # sample
            if (it + 1) % 100 == 0:
                x_sample_opt_list = [xa_sample_ipt, np.full((n_sample, img_size, img_size // 10, 3), -1.0)]
                for i, b_sample_ipt in enumerate(b_sample_ipt_list):
                    _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
                    if i > 0:   # i == 0 is for reconstruction
                        _b_sample_ipt[..., i - 1] = _b_sample_ipt[..., i - 1] * test_int / thres_int
                    x_sample_opt_list.append(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt, _b_sample: _b_sample_ipt}))
                sample = np.concatenate(x_sample_opt_list, 2)

                save_dir = './output/%s/sample_training' % experiment_name
                pylib.mkdir(save_dir)
                im.imwrite(im.immerge(sample, n_sample, 1), '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, it_in_epoch, it_per_epoch))

except:
    traceback.print_exc()
finally:
    save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_in_epoch, it_per_epoch))
    print('Model is saved at %s!' % save_path)
    sess.close()
