"""
Name:        "plot_results.py"
Author:      Marco Olivieri
Date:        23 November 2021
Description: Script to plot information and results
"""

import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_dir, "../"))  # to import other module

import matplotlib.pyplot as plt
import numpy as np

vmin_abs = 0
vmax_abs = 1
vmin_angle = -np.pi
vmax_angle = np.pi
my_cmap = plt.cm.get_cmap("jet")

def plot_abs_phase(pred, gt, quantity, index, size):
    """
    Plot the abs and phase part of prediction and ground truth
    :param pred: prediction dataset
    :param gt: groundtruth dataset
    :param quantity: string specifying the physics
    :param index: idx of dataset to plot
    :param size: size of the pressure image
    :return:
    """
    plt.figure()
    plt.subplot(221)
    plt.imshow(np.abs(pred[index]).reshape(size[0], size[1]), vmin=vmin_abs, vmax=vmax_abs, cmap=my_cmap), plt.colorbar()
    plt.title("abs_" + quantity + "_hat")
    plt.subplot(222)
    plt.imshow(np.abs(gt[index]).reshape(size[0], size[1]), vmin=vmin_abs, vmax=vmax_abs, cmap=my_cmap), plt.colorbar()
    plt.title("abs_" + quantity + "_gt")
    plt.subplot(223)
    plt.imshow(np.angle(pred[index]).reshape(size[0], size[1]), vmin=vmin_angle, vmax=vmax_angle, cmap=my_cmap), plt.colorbar()
    plt.title("phase_" + quantity + "_hat")
    plt.subplot(224)
    plt.imshow(np.angle(gt[index]).reshape(size[0], size[1]), vmin=vmin_angle, vmax=vmax_angle, cmap=my_cmap), plt.colorbar()
    plt.title("phase_" + quantity + "_gt")
    plt.show()


def print_information(i, mode_test=[], freq_test=[], num_plate_test=[], boundary_test=[], plate_dimensions_test=[],
                      snr_test=[]):
    """
    Print and return all the information relate to a specified test case. If pass "" -> not print that information
    :param i: index to print information
    :param all the test set infotmation
    :return: information
    """
    information = ""
    if len(boundary_test) != 0:
        information += f"bound:\t{boundary_test[i]}\n"
    if len(freq_test) != 0:
        information += f"freq:\t{np.round(freq_test[i], 2)} Hz\n"
    if len(mode_test) != 0:
        information += f"mode:\t{mode_test[i]}\n"
    if len(num_plate_test) != 0:
        information += f"plate:\t{num_plate_test[i]}\n"
    if len(plate_dimensions_test) != 0:
        information += f"dim:\t{plate_dimensions_test[i]} m\n"
    if len(snr_test) != 0:
        information += f"snr:\t{snr_test[i]} dB\n"

    information += f"index:\t{i}\n"

    print(information)
    return information
