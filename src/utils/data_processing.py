"""
Name:        "data_processing.py"
Author:      Marco Olivieri
Date:        23 November 2021
Description: General data processing
"""

import numpy as np
import tensorflow as tf

def normalise_wrt_abs(complex_data):
    """
    normalized complex_data wrt to max(abs)
    :param complex_data: numpy array
    :return:
    """
    data = complex_data.copy()
    data = data.reshape(data.shape[0], -1)
    M = np.max(abs(data), axis=-1).reshape(-1, 1)  # maximum for each images
    M = np.tile(M, (1, data.shape[-1]))
    data_norm = data / M

    data_norm = data_norm.reshape(complex_data.shape)

    return data_norm


def normalise_wrt_abs_tf(complex_data):
    """
    normalized complex_data wrt to max(abs)
    :param complex_data: tensorflow array
    :return: data_norm
    """

    data = complex_data
    img_size = [complex_data.shape[1], complex_data.shape[2]]

    # check shape (if real and imag splitted)
    if complex_data.shape[-1] == 2:
        # merge real and imag values
        data = tf.complex(data[:,:,:,0], data[:,:,:,1])

    # normalization
    data = tf.reshape(data, [-1, img_size[0]*img_size[1]])
    data = tf.cast(data, tf.complex128)
    M = tf.reduce_max(abs(data), axis=-1)
    M = tf.reshape(M, [-1, 1])  # maximum for each images
    M = tf.tile(M, (1, data.shape[-1]))
    M = tf.cast(M, tf.complex128)
    data_norm = data / M

    data_norm = tf.reshape(data_norm, [-1, img_size[0], img_size[1]])

    # check shape (if real and imag splitted)
    if complex_data.shape[-1] == 2:
        # merge real and imag values
        data_norm = tf.expand_dims(data_norm, axis=-1)
        data_norm = tf.concat((tf.math.real(data_norm), tf.math.imag(data_norm)), axis=-1)

    return data_norm


def split_real_imag_channels(data):
    """
    Split complex image in two channels
    :param data: complex data
    :return: out_channels - data in two channels (real and imaginary parts)
    """
    real_img = np.copy(np.real(data))
    imag_img = np.copy(np.imag(data))

    out_channels = np.concatenate((real_img, imag_img), axis=-1)

    return out_channels
