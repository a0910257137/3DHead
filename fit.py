import tensorflow as tf
import cv2
import numpy as np
import os
from monitor import logger
import core.utils as utils
from core.factory import ModelFactory
import argparse
from box import Box
from utils.sys_tools import set_gpu
from utils.tools import load_configger


def fit(config):
    # step 1. By using model inference, bbox and  landmarks
    # step 2. cropped image
    resized_face_img = cv2.imread("./test/output.jpg")[..., ::-1]
    resized_face_img = tf.cast(resized_face_img, tf.float32)
    lms = np.load("./test/lms.npy")
    lms = lms[:, :2][None, ...]
    lms = tf.cast(lms, tf.float32)
    lm_weights = utils.get_lm_weights()

    print('start rigid fitting')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    recon_model = ModelFactory(config.model).build_model()
    coeffs = recon_model._packed_tensors()
    recon_model(coeffs, True)

    return


def argparser():
    parser = argparse.ArgumentParser(description='3DMM')
    parser.add_argument('--gpus', help="Selec gpu id")
    parser.add_argument('--config', type=str, help="Configure file")
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()
    set_gpu(args.gpus)
    logger.info(f'Use config: {args.config} to fit 3D head')
    config = Box(load_configger(args.config))
    fit(config)