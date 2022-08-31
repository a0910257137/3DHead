import numpy as np
import torch
import torch.nn.functional as F
import tensorflow as tf


def photo_loss(pred_img, gt_img, img_mask):
    img_mask = tf.cast(img_mask, tf.float32)
    loss = tf.math.sqrt(
        tf.math.reduce_sum(tf.square(pred_img - gt_img),
                           axis=3)) * img_mask / 255

    loss = tf.math.reduce_sum(loss, axis=[1, 2]) / tf.math.reduce_sum(
        img_mask, axis=[1, 2])
    loss = tf.math.reduce_mean(loss)
    return loss


def lm_loss(pred_lms, gt_lms, weight, img_size=256):

    loss = tf.math.reduce_sum(
        tf.math.square(pred_lms / img_size - gt_lms / img_size),
        axis=2) * tf.reshape(tf.cast(weight, tf.float32), (1, -1))

    loss = tf.math.reduce_mean(tf.math.reduce_sum(loss, axis=1))
    return loss


def get_l2(tensor):
    return tf.math.reduce_sum(tf.math.square(tensor))


def reflectance_loss(tex, skin_mask):
    skin_mask = tf.expand_dims(skin_mask, axis=-1)
    tex_mean = tf.math.reduce_sum(
        tex * skin_mask, axis=1, keepdims=True) / tf.math.reduce_sum(skin_mask)
    loss =  tf.math.reduce_sum(tf.math.square(tex - tex_mean) * skin_mask / 255.)/ \
        (tf.cast(tf.shape(tex)[0], tf.float32) * tf.math.reduce_sum(skin_mask))

    return loss


def gamma_loss(gamma):

    gamma = gamma.reshape(-1, 3, 9)
    gamma_mean = torch.mean(gamma, dim=1, keepdims=True)
    gamma_loss = torch.mean(torch.square(gamma - gamma_mean))

    return gamma_loss
