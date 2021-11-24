"""
Name:        "khcnn_network.py"
Author:      Marco Olivieri
Date:        23 November 2021
Description: KHCNN complete architecture
"""

import os
from utils import params as PARAMS

os.environ['CUDA_VISIBLE_DEVICES'] = PARAMS.GPUS
os.environ['CUDA_ALLOW_GROWTH'] = 'True'

import tensorflow as tf
from utils import data_processing as DATA
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from network_module import cnn_models as CNN
from network_module import kh_discretisation_tf as KH


def build_khcnn_model(p_size, s_size):
    """
    Build the KHCNN model
    :param p_size: pressure resolution
    :param s_size: surface resolution
    :return: keras model, losses dict, lossWeights dict
    """
    p_h_input_imgs = Input(shape=(p_size[0], p_size[1], 2))  # channels = real & imag of pressure
    freqs = Input(shape=(1))                            # [None, 1]
    h_points = Input(shape=(p_size[0] * p_size[1], 3))  # [None, #points (1024 or 64), 3]
    s_points = Input(shape=(s_size[0] * s_size[1], 3))  # [None, 1024, 3]
    bm = Input(shape=(s_size[0], s_size[1], 2))

    # 2 outputs of shape=(16, 64, 2) - channles = real & imag
    # full structure KH
    if PARAMS.USE_LOW_P_RESOLUTION:
        v_s_output_imgs, p_s_output_imgs = CNN.cnn_model_64(p_h_input_imgs, start_neurons=4, out_channel=2,
                                                            name="complex_64")  # num = start_neurons
    else:
        v_s_output_imgs, p_s_output_imgs = CNN.cnn_model_1024(p_h_input_imgs, start_neurons=8, out_channel=2,
                                                              name="complex_1024")  # num = start_neurons

    v_s_output_imgs = tf.math.multiply(v_s_output_imgs, bm, name="v_s_masked")
    p_s_output_imgs = tf.math.multiply(p_s_output_imgs, bm, name="v_s_masked")

    real_p_h_hat, imag_p_h_hat = KH.forward_propagation(v_s_output_imgs, p_s_output_imgs, freqs, h_points, s_points)

    # -----------------

    # loss on abs & phase
    v_s_hat = tf.complex(v_s_output_imgs[:, :, :, 0], v_s_output_imgs[:, :, :, 1])
    p_h_hat = tf.complex(real_p_h_hat, imag_p_h_hat)
    p_h_hat = DATA.normalise_wrt_abs_tf(p_h_hat)    # normalized wrt max(abs)

    model = Model(inputs=[p_h_input_imgs, freqs, h_points, s_points, bm],
                  outputs=[tf.math.real(p_h_hat), tf.math.imag(p_h_hat),
                           tf.math.real(v_s_hat), tf.math.imag(v_s_hat),
                           ])

    model.output_names = ["real_p_h_hat", "imag_p_h_hat",
                          "real_v_s_hat", "imag_v_s_hat"
                          ]

    losses = {
        "real_p_h_hat": "mean_squared_error",
        "imag_p_h_hat": "mean_squared_error",
        "real_v_s_hat": "mean_squared_error",
        "imag_v_s_hat": "mean_squared_error"
    }

    lossWeights = {
        "real_p_h_hat": 0.5, "imag_p_h_hat": 0.5,
        "real_v_s_hat": 0.5, "imag_v_s_hat": 0.5
    }

    return model, losses, lossWeights