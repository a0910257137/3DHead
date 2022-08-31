import tensorflow as tf
import cv2
import numpy as np
import os
import core.utils as utils
import argparse
from monitor import logger
from core.factory import ModelFactory
from box import Box
from utils.sys_tools import set_gpu
from utils.tools import load_configger
from tqdm import tqdm
from core.losses import *


def fit(config):
    face_img = cv2.imread("./test/output.jpg")[..., ::-1]
    face_h, face_w, channel = face_img.shape
    resized_face_img = cv2.resize(face_img, (256, 256))
    resized_face_img = tf.cast(resized_face_img, tf.float32)
    lms = np.load("./test/lms.npy")
    lms = lms[:, :2][None, ...]
    lms = tf.cast(lms, tf.float32)
    img_tensor = resized_face_img[None, ...]
    lm_weights = utils.get_lm_weights()

    print('start rigid fitting')
    recon_model = ModelFactory(config.model).build_model()
    first_rf_iters = 500
    first_nrf_iters = 500
    lm_loss_w = 100
    id_reg_w = 0.001
    exp_reg_w = 0.0008
    tex_reg_w = 1.7e-06
    tex_w = 1
    rgb_loss_w = 1.6
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    recon_model.build(input_shape=[])
    for i in tqdm(range(first_rf_iters)):
        with tf.GradientTape() as tape:
            pred_dict = recon_model(render=False)
            lm_loss_val = lm_loss(pred_dict['lms_proj'],
                                  lms,
                                  lm_weights,
                                  img_size=config.model.img_size[0])
            total_loss = lm_loss_w * lm_loss_val
        rot_tensor = recon_model.get_rot_tensor()
        trans_tensor = recon_model.get_trans_tensor()
        grads = tape.gradient(total_loss, [rot_tensor, trans_tensor])
        optimizer.apply_gradients(zip(grads, [rot_tensor, trans_tensor]))
        print('-' * 100)
        print(rot_tensor)
    
    non_rigid_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    for i in tqdm(range(first_nrf_iters)):
        with tf.GradientTape() as tape:
            pred_dict = recon_model(render=True)
            rendered_img = pred_dict['rendered_img']
            lms_proj = pred_dict['lms_proj']
            face_texture = pred_dict['face_texture']
            mask = rendered_img[:, :, :, 3]
            photo_loss_val = photo_loss(rendered_img[:, :, :, :3], img_tensor,
                                        mask > 0)

            lm_loss_val = lm_loss(lms_proj,
                                  lms,
                                  lm_weights,
                                  img_size=config.model.img_size[0])
            id_reg_loss = get_l2(recon_model.get_id_tensor())
            exp_reg_loss = get_l2(recon_model.get_exp_tensor())
            tex_reg_loss = get_l2(recon_model.get_tex_tensor())
            tex_loss_val = reflectance_loss(face_texture,
                                            recon_model.get_skinmask())
            loss = lm_loss_val * lm_loss_w + id_reg_loss * id_reg_w + exp_reg_loss * exp_reg_w + tex_reg_loss * tex_reg_w + tex_loss_val * tex_w + photo_loss_val * rgb_loss_w

        id_tensor = recon_model.get_id_tensor()
        exp_tensor = recon_model.get_exp_tensor()
        gamma_tensor = recon_model.get_gamma_tensor()
        tex_tensor = recon_model.get_tex_tensor()
        rot_tensor = recon_model.get_rot_tensor()
        trans_tensor = recon_model.get_trans_tensor()
        grads = tape.gradient(loss, [
            id_tensor, exp_tensor, gamma_tensor, tex_tensor, rot_tensor,
            trans_tensor
        ])
        non_rigid_optimizer.apply_gradients(
            zip(grads, [
                id_tensor, exp_tensor, gamma_tensor, tex_tensor, rot_tensor,
                trans_tensor
            ]))

    loss_str = ''
    loss_str += 'lm_loss: %f\t' % lm_loss_val
    loss_str += 'photo_loss: %f\t' % photo_loss_val
    loss_str += 'tex_loss: %f\t' % tex_loss_val
    loss_str += 'id_reg_loss: %f\t' % id_reg_loss
    loss_str += 'exp_reg_loss: %f\t' % exp_reg_loss
    loss_str += 'tex_reg_loss: %f\t' % tex_reg_loss
    print('done non rigid fitting.', loss_str)

    pred_dict = recon_model(render=True)
    rendered_img = tf.squeeze(pred_dict['rendered_img']).numpy()
    out_img = rendered_img[:, :, :3].astype(np.uint8)
    out_mask = (rendered_img[:, :, 3] > 0).astype(np.uint8)
    resized_out_img = cv2.resize(out_img, (face_w, face_h))
    resized_mask = cv2.resize(out_mask, (face_w, face_h),
                              cv2.INTER_NEAREST)[..., None]

    img_arr = cv2.imread(
        "/home2/anders/proj_py/3DHead/test/000002.jpg")[:, :, ::-1]
    # composed_img = img_arr.copy()
    composed_face = face_img * (1 -
                                resized_mask) + resized_out_img * resized_mask
    cv2.imwrite("output.jpg", composed_face[..., ::-1])


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