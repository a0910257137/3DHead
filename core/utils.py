from asyncio import constants
import tensorflow as tf


def get_lm_weights():
    w = tf.ones(shape=(68, 1), dtype=tf.int32)

    index1 = tf.constant([[28], [29], [30]])
    val1 = tf.constant([10], shape=(3, 1))
    index2 = tf.constant(tf.range(48, 68), shape=(68 - 48, 1))
    val2 = tf.constant([10], shape=(68 - 48, 1))

    w = tf.tensor_scatter_nd_update(w, index1, val1)
    w = tf.tensor_scatter_nd_update(w, index2, val2)
    norm_w = w / tf.math.reduce_sum(w)
    return norm_w