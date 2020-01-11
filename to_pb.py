import pylib as py
import tensorflow as tf
import tflib as tl

import module

from tensorflow.python.framework import graph_util


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_name', default='default')
args_ = py.args()

# output_dir
output_dir = py.join('output', args_.experiment_name)

# save settings
args = py.args_from_yaml(py.join(output_dir, 'settings.yml'))
args.__dict__.update(args_.__dict__)

# others
n_atts = len(args.att_names)

sess = tl.session()
sess.__enter__()  # make default


# ==============================================================================
# =                                   graph                                    =
# ==============================================================================

def sample_graph():
    # model
    Genc, Gdec, _ = module.get_model(args.model, n_atts, weight_decay=args.weight_decay)

    # placeholders & inputs
    xa = tf.placeholder(tf.float32, shape=[None, args.crop_size, args.crop_size, 3], name='xa')
    b_ = tf.placeholder(tf.float32, shape=[None, n_atts], name='b_')

    # sample graph
    x = Gdec(Genc(xa, training=False), b_, training=False)
    x = tf.identity(x, name='xb')


sample = sample_graph()


# ==============================================================================
# =                                   freeze                                   =
# ==============================================================================

# checkpoint
checkpoint = tl.Checkpoint(
    {v.name: v for v in tf.global_variables()},
    py.join(output_dir, 'checkpoints'),
    max_to_keep=1
)
checkpoint.restore().run_restore_ops()

with tf.gfile.GFile(py.join(output_dir, 'generator.pb'), 'wb') as f:
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['xb'])
    f.write(constant_graph.SerializeToString())

sess.close()
