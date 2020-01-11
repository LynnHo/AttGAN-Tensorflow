import tensorflow as tf


def center_loss(features, labels, num_classes, alpha=0.5, updates_collections=tf.GraphKeys.UPDATE_OPS, scope=None):
    # modified from https://github.com/EncodeTS/TensorFlow_Center_Loss/blob/master/center_loss.py

    assert features.shape.ndims == 2, 'The rank of `features` should be 2!'
    assert 0 <= alpha <= 1, '`alpha` should be in [0, 1]!'

    with tf.variable_scope(scope, 'center_loss', [features, labels]):
        centers = tf.get_variable('centers', shape=[num_classes, features.get_shape()[-1]], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)

        centers_batch = tf.gather(centers, labels)
        diff = centers_batch - features
        _, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff
        update_centers = tf.scatter_sub(centers, labels, diff)

        center_loss = 0.5 * tf.reduce_mean(tf.reduce_sum((centers_batch - features)**2, axis=-1))

        if updates_collections is None:
            with tf.control_dependencies([update_centers]):
                center_loss = tf.identity(center_loss)
        else:
            tf.add_to_collections(updates_collections, update_centers)

    return center_loss, centers


def sigmoid_focal_loss(multi_class_labels, logits, gamma=2.0):
    epsilon = 1e-8
    multi_class_labels = tf.cast(multi_class_labels, logits.dtype)

    p = tf.sigmoid(logits)
    pt = p * multi_class_labels + (1 - p) * (1 - multi_class_labels)
    focal_loss = tf.reduce_mean(- (1 - pt)**gamma * tf.log(tf.maximum(pt, epsilon)))

    return focal_loss
