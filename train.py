import traceback

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tflib as tl
import tfprob
import tqdm

import data
import module


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

default_att_names = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
                     'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
py.arg('--att_names', choices=data.ATT_ID.keys(), nargs='+', default=default_att_names)

py.arg('--img_dir', default='./data/img_celeba/aligned/align_size(572,572)_move(0.250,0.000)_face_factor(0.450)_jpg/data')
py.arg('--train_label_path', default='./data/img_celeba/train_label.txt')
py.arg('--val_label_path', default='./data/img_celeba/val_label.txt')
py.arg('--load_size', type=int, default=143)
py.arg('--crop_size', type=int, default=128)

py.arg('--n_epochs', type=int, default=60)
py.arg('--epoch_start_decay', type=int, default=30)
py.arg('--batch_size', type=int, default=32)
py.arg('--learning_rate', type=float, default=2e-4)
py.arg('--beta_1', type=float, default=0.5)

py.arg('--model', default='model_128', choices=['model_128', 'model_256', 'model_384'])

py.arg('--n_d', type=int, default=5)  # # d updates per g update
py.arg('--adversarial_loss_mode', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'], default='wgan')
py.arg('--gradient_penalty_mode', choices=['none', '1-gp', '0-gp', 'lp'], default='1-gp')
py.arg('--gradient_penalty_sample_mode', choices=['line', 'real', 'fake', 'dragan'], default='line')
py.arg('--d_gradient_penalty_weight', type=float, default=10.0)
py.arg('--d_attribute_loss_weight', type=float, default=1.0)
py.arg('--g_attribute_loss_weight', type=float, default=10.0)
py.arg('--g_reconstruction_loss_weight', type=float, default=100.0)
py.arg('--weight_decay', type=float, default=0.0)

py.arg('--n_samples', type=int, default=12)
py.arg('--test_int', type=float, default=2.0)

py.arg('--experiment_name', default='default')
args = py.args()

# output_dir
output_dir = py.join('output', args.experiment_name)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# others
n_atts = len(args.att_names)

sess = tl.session()
sess.__enter__()  # make default


# ==============================================================================
# =                               data and model                               =
# ==============================================================================

# data
train_dataset, len_train_dataset = data.make_celeba_dataset(args.img_dir, args.train_label_path, args.att_names, args.batch_size,
                                                            load_size=args.load_size, crop_size=args.crop_size,
                                                            training=True, shuffle=True, repeat=None)
val_dataset, len_val_dataset = data.make_celeba_dataset(args.img_dir, args.val_label_path, args.att_names, args.n_samples,
                                                        load_size=args.load_size, crop_size=args.crop_size,
                                                        training=False, shuffle=True, repeat=None)
train_iter = train_dataset.make_one_shot_iterator()
val_iter = val_dataset.make_one_shot_iterator()

# model
Genc, Gdec, D = module.get_model(args.model, n_atts, weight_decay=args.weight_decay)

# loss functions
d_loss_fn, g_loss_fn = tfprob.get_adversarial_losses_fn(args.adversarial_loss_mode)


# ==============================================================================
# =                                   graph                                    =
# ==============================================================================

def D_train_graph():
    # ======================================
    # =               graph                =
    # ======================================

    # placeholders & inputs
    lr = tf.placeholder(dtype=tf.float32, shape=[])

    xa, a = train_iter.get_next()
    b = tf.random_shuffle(a)
    b_ = b * 2 - 1

    # generate
    z = Genc(xa)
    xb_ = Gdec(z, b_)

    # discriminate
    xa_logit_gan, xa_logit_att = D(xa)
    xb__logit_gan, xb__logit_att = D(xb_)

    # discriminator losses
    xa_loss_gan, xb__loss_gan = d_loss_fn(xa_logit_gan, xb__logit_gan)
    gp = tfprob.gradient_penalty(lambda x: D(x)[0], xa, xb_, args.gradient_penalty_mode, args.gradient_penalty_sample_mode)
    xa_loss_att = tf.losses.sigmoid_cross_entropy(a, xa_logit_att)
    reg_loss = tf.reduce_sum(D.func.reg_losses)

    loss = (xa_loss_gan + xb__loss_gan +
            gp * args.d_gradient_penalty_weight +
            xa_loss_att * args.d_attribute_loss_weight +
            reg_loss)

    # optim
    step_cnt, _ = tl.counter()
    step = tf.train.AdamOptimizer(lr, beta1=args.beta_1).minimize(loss, global_step=step_cnt, var_list=D.func.trainable_variables)

    # summary
    with tf.contrib.summary.create_file_writer('./output/%s/summaries/D' % args.experiment_name).as_default(),\
            tf.contrib.summary.record_summaries_every_n_global_steps(10, global_step=step_cnt):
        summary = [
            tl.summary_v2({
                'loss_gan': xa_loss_gan + xb__loss_gan,
                'gp': gp,
                'xa_loss_att': xa_loss_att,
                'reg_loss': reg_loss
            }, step=step_cnt, name='D'),
            tl.summary_v2({'lr': lr}, step=step_cnt, name='learning_rate')
        ]

    # ======================================
    # =            run function            =
    # ======================================

    def run(**pl_ipts):
        sess.run([step, summary], feed_dict={lr: pl_ipts['lr']})

    return run


def G_train_graph():
    # ======================================
    # =                 graph              =
    # ======================================

    # placeholders & inputs
    lr = tf.placeholder(dtype=tf.float32, shape=[])

    xa, a = train_iter.get_next()
    b = tf.random_shuffle(a)
    a_ = a * 2 - 1
    b_ = b * 2 - 1

    # generate
    z = Genc(xa)
    xa_ = Gdec(z, a_)
    xb_ = Gdec(z, b_)

    # discriminate
    xb__logit_gan, xb__logit_att = D(xb_)

    # generator losses
    xb__loss_gan = g_loss_fn(xb__logit_gan)
    xb__loss_att = tf.losses.sigmoid_cross_entropy(b, xb__logit_att)
    xa__loss_rec = tf.losses.absolute_difference(xa, xa_)
    reg_loss = tf.reduce_sum(Genc.func.reg_losses + Gdec.func.reg_losses)

    loss = (xb__loss_gan +
            xb__loss_att * args.g_attribute_loss_weight +
            xa__loss_rec * args.g_reconstruction_loss_weight +
            reg_loss)

    # optim
    step_cnt, _ = tl.counter()
    step = tf.train.AdamOptimizer(lr, beta1=args.beta_1).minimize(loss, global_step=step_cnt, var_list=Genc.func.trainable_variables + Gdec.func.trainable_variables)

    # summary
    with tf.contrib.summary.create_file_writer('./output/%s/summaries/G' % args.experiment_name).as_default(),\
            tf.contrib.summary.record_summaries_every_n_global_steps(10, global_step=step_cnt):
        summary = tl.summary_v2({
            'xb__loss_gan': xb__loss_gan,
            'xb__loss_att': xb__loss_att,
            'xa__loss_rec': xa__loss_rec,
            'reg_loss': reg_loss
        }, step=step_cnt, name='G')

    # ======================================
    # =           generator size           =
    # ======================================

    n_params, n_bytes = tl.count_parameters(Genc.func.variables + Gdec.func.variables)
    print('Generator Size: n_parameters = %d = %.2fMB' % (n_params, n_bytes / 1024 / 1024))

    # ======================================
    # =            run function            =
    # ======================================

    def run(**pl_ipts):
        sess.run([step, summary], feed_dict={lr: pl_ipts['lr']})

    return run


def sample_graph():
    # ======================================
    # =               graph                =
    # ======================================

    # placeholders & inputs
    val_next = val_iter.get_next()
    xa = tf.placeholder(tf.float32, shape=[None, args.crop_size, args.crop_size, 3])
    b_ = tf.placeholder(tf.float32, shape=[None, n_atts])

    # sample graph
    x = Gdec(Genc(xa, training=False), b_, training=False)

    # ======================================
    # =            run function            =
    # ======================================

    save_dir = './output/%s/samples_training' % args.experiment_name
    py.mkdir(save_dir)

    def run(epoch, iter):
        # data for sampling
        xa_ipt, a_ipt = sess.run(val_next)
        b_ipt_list = [a_ipt]  # the first is for reconstruction
        for i in range(n_atts):
            tmp = np.array(a_ipt, copy=True)
            tmp[:, i] = 1 - tmp[:, i]   # inverse attribute
            tmp = data.check_attribute_conflict(tmp, args.att_names[i], args.att_names)
            b_ipt_list.append(tmp)

        x_opt_list = [xa_ipt]
        for i, b_ipt in enumerate(b_ipt_list):
            b__ipt = (b_ipt * 2 - 1).astype(np.float32)  # !!!
            if i > 0:   # i == 0 is for reconstruction
                b__ipt[..., i - 1] = b__ipt[..., i - 1] * args.test_int
            x_opt = sess.run(x, feed_dict={xa: xa_ipt, b_: b__ipt})
            x_opt_list.append(x_opt)
        sample = np.transpose(x_opt_list, (1, 2, 0, 3, 4))
        sample = np.reshape(sample, (-1, sample.shape[2] * sample.shape[3], sample.shape[4]))
        im.imwrite(sample, '%s/Epoch-%d_Iter-%d.jpg' % (save_dir, epoch, iter))

    return run


D_train_step = D_train_graph()
G_train_step = G_train_graph()
sample = sample_graph()


# ==============================================================================
# =                                   train                                    =
# ==============================================================================

# step counter
step_cnt, update_cnt = tl.counter()

# checkpoint
checkpoint = tl.Checkpoint(
    {v.name: v for v in tf.global_variables()},
    py.join(output_dir, 'checkpoints'),
    max_to_keep=1
)
checkpoint.restore().initialize_or_restore()

# summary
sess.run(tf.contrib.summary.summary_writer_initializer_op())

# learning rate schedule
lr_fn = tl.LinearDecayLR(args.learning_rate, args.n_epochs, args.epoch_start_decay)

# train
try:
    for ep in tqdm.trange(args.n_epochs, desc='Epoch Loop'):
        # learning rate
        lr_ipt = lr_fn(ep)

        for it in tqdm.trange(len_train_dataset, desc='Inner Epoch Loop'):
            if it + ep * len_train_dataset < sess.run(step_cnt):
                continue
            step = sess.run(update_cnt)

            # train D
            if step % (args.n_d + 1) != 0:
                D_train_step(lr=lr_ipt)
            # train G
            else:
                G_train_step(lr=lr_ipt)

            # save
            if step % (1000 * (args.n_d + 1)) == 0:
                checkpoint.save(step)

            # sample
            if step % (100 * (args.n_d + 1)) == 0:
                sample(ep, it)
except Exception:
    traceback.print_exc()
finally:
    checkpoint.save(step)
    sess.close()
