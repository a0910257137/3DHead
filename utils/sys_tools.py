import tensorflow as tf
import os


def set_gpu(gpu_ids):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for i in range(len(gpus)):
        tf.config.experimental.set_memory_growth(gpus[i], True)
