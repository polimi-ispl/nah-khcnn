"""
Name:        "metrics.py"
Author:      Marco Olivieri
Date:        23 November 2021
Description: Complex metrics
"""

import numpy as np


def nmse(x_hat, x):
    """
    Compute NMSE in complex number of a single image
    :param x_hat: prediction
    :param x: groundtruth
    :return: nmse dB
    """
    x = x.flatten()
    x_hat = x_hat.flatten()
    err = x_hat - x
    nmse = np.sum(err * np.conj(err)) / np.sum(x * np.conj(x))
    nmse = nmse.real  # imaginary part is zero
    return 10 * np.log10(nmse)


def ncc(x_hat, x):
    """
    Compute NCC in complex number of a single image
    :param x_hat: prediction
    :param x: groundtruth
    :return: ncc
    """
    x_hat = x_hat.flatten()
    x = x.flatten()

    num = abs(np.sum(x_hat * np.conj(x)))
    den = np.linalg.norm(x_hat) * np.linalg.norm(x)

    return num/den
