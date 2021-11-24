"""
Name:        "kh_discretisation_tf.py"
Author:      Marco Olivieri
Date:        23 November 2021
Description: Kirchhoff–Helmholtz discretisation script
"""

import os
from utils import params as PARAMS

os.environ['CUDA_VISIBLE_DEVICES'] = PARAMS.GPUS
os.environ['CUDA_ALLOW_GROWTH'] = 'True'

import tensorflow as tf
import math
import numpy as np

def forward_propagation(v_s_imgs, p_s_imgs, freqs, h_points, s_points):
    """
    Compute the KH forward propagation
    :param v_s_imgs: estimated surface velocity
    :param p_s_imgs: estimated surface pressure
    :param freqs: vibrational frequencies
    :param h_points: hologram mesh points
    :param s_points: surface mesh points
    :return: real_p_h_hat, imag_p_h_hat: real and imaginary estimation of hologram pressure
    """

    v_s = tf.complex(v_s_imgs[:, :, :, 0], v_s_imgs[:, :, :, 1])
    v_s = tf.reshape(v_s, [-1, v_s.shape[1] * v_s.shape[2]])

    p_s = tf.complex(p_s_imgs[:, :, :, 0], p_s_imgs[:, :, :, 1])
    p_s = tf.reshape(p_s, [-1, p_s.shape[1] * p_s.shape[2]])

    omegas = 2 * np.pi * freqs

    p_h_hat = compute_kh_discrete(h_points, s_points, p_s, v_s, omegas)

    p_size = (16, 64)
    if PARAMS.USE_LOW_P_RESOLUTION:
        p_size = (8, 8)
    real_p_h_hat = tf.reshape(tf.math.real(p_h_hat), [-1, p_size[0], p_size[1]])
    imag_p_h_hat = tf.reshape(tf.math.imag(p_h_hat), [-1, p_size[0], p_size[1]])

    return real_p_h_hat, imag_p_h_hat


def compute_kh_discrete(h_points, s_points, p_s, v_s, omega):
    """
    Compute the discretized Kirchhoff-Helmholtz integral equation with Riemann sum
    :param h_points: hologram mesh points
    :param s_points: surface mesh points
    :param p_s: estimated surface pressure
    :param v_s: estimated surface velocity
    :param omega: vibrational angular frequency
    :return: p_h_hat: complex hologram pressure estimation
    """

    num_s_points = s_points.shape[1]
    num_h_points = h_points.shape[1]

    stepX_s, stepY_s = get_step_size(s_points, shapeImg=(16, 64))

    z_h = h_points[:, None, 0, 2] - s_points[:, None, 0, 2]

    # get elevation distance for each points
    z = tf.expand_dims(h_points[:, :, 2], axis=-1)
    z = tf.repeat(z, num_s_points, axis=-1)  # z holographic duplicate for each surface points
    z0 = tf.expand_dims(s_points[:, :, 2], axis=-1)
    z0 = tf.transpose(tf.repeat(z0, num_h_points, axis=-1), [0, 2, 1])  # z surface duplicate for each holographic points and transpose
    z_z0 = z - z0

    # get point distances
    dist = pairwise_dist(h_points, s_points)

    # get green functions
    green = get_green_functions(omega, dist)
    d_green = get_derivative_green_functions(omega, dist, z_z0)

    d_green = tf.cast(d_green, tf.complex128)
    p_s = tf.expand_dims(p_s, axis=-1)

    # integrand 1 - compute matrix multiplication
    tmp = tf.transpose(p_s, (0, 2, 1))
    tmp = tf.cast(tf.tile(tmp, multiples=[1, num_h_points, 1]), tf.complex128)
    i1 = d_green * tmp
    i1 = tf.reduce_sum(i1, axis=2)

    # integrand 2 - compute matrix multiplication
    green = tf.cast(green, tf.complex128)
    v_s = tf.expand_dims(v_s, axis=-1)
    tmp = tf.transpose(v_s, (0, 2, 1))
    tmp = tf.cast(tf.tile(tmp, multiples=[1, num_h_points, 1]), tf.complex128)
    i2 = green * tmp
    i2 = tf.reduce_sum(i2, axis=2)

    stepes = stepX_s * stepY_s
    stepes = tf.cast(stepes, tf.complex128)

    # KH integral
    p_h_hat = -(i1 - 1j * tf.cast(omega * PARAMS.rho_0, tf.complex128) * i2) * stepes

    return p_h_hat


def get_step_size(points, shapeImg):
    """
    Compute the step size from a regular grid points
    :param
        points:   grid point in vector
        shapeImg: image resolution
    :return:
        step size of sampling
    """

    tmp = tf.reshape(points, [-1, *shapeImg, 3])
    stepX = tmp[:, None, 0, 1, 0] - tmp[:, None, 0, 0, 0]
    stepY = tmp[:, None, 1, 0, 1] - tmp[:, None, 0, 0, 1]
    return stepX, stepY


def pairwise_dist(A, B):
    """
    Computes pairwise distances between each elements of A and each elements of B.
    Args:
      A,    [m,d] matrix
      B,    [n,d] matrix
    Returns:
      D,    [m,n] matrix of pairwise distances
    """

    na = tf.reduce_sum(tf.square(A), 2)
    nb = tf.reduce_sum(tf.square(B), 2)

    na = tf.reshape(na, [-1, na.shape[-1], 1])
    nb = tf.reshape(nb, [-1, 1, nb.shape[-1]])

    # return pairwise euclidead difference matrix
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))

    return D


def get_green_functions(omega, distances):
    """
    Compute the Green's function in free field
    :param
        omega:      angular frequency for which to compute the Green function
        distances:  distances matrix (#mic points x #surf points)
    :return:
        matrix (#mic points x #surf points) with the green function for each pair of point (r, s)
    """
    k = omega/PARAMS.c  # wavenumber
    k = tf.expand_dims(k, axis=-1)
    h_points = distances.shape[-2]
    s_points = distances.shape[-1]
    K = tf.tile(k, multiples=[1, h_points, s_points])

    numerator = tf.math.exp(tf.complex(0.0, -K * distances))  # tile
    denominator = tf.cast(4 * math.pi * distances, dtype=tf.complex64)
    green = numerator/denominator

    return green


def get_derivative_green_functions(omega, distances, z_h):
    """
    Compute the Green's function derivatives in free field (and consider planar plates)
    :param
        omega:      angular frequency for which to compute the Green function
        distances:  distances matrix (#mic points x #surf points)
        z_h:        height of the hologram plane (distance from hologram and surface)
    :return:
        matrix (#mic points x #surf points) with the green function derivative for each pair of point (r, s)
    """
    k = omega/PARAMS.c  # wavenumber
    k = tf.expand_dims(k, axis=-1)
    h_points = distances.shape[-2]
    s_points = distances.shape[-1]
    K = tf.tile(k, multiples=[1, h_points, s_points])

    Z_h = z_h
    numerator = -tf.math.exp(tf.complex(0.0, -K * distances)) * tf.cast(Z_h, tf.complex64) * tf.complex(1.0, K * distances)
    denominator = tf.cast(4 * math.pi * (distances ** 3), dtype=tf.complex64)
    d_greens = numerator / denominator

    return d_greens
