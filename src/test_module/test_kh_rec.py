"""
Name:        "test_kh_rec.py"
Author:      Marco Olivieri
Date:        23 November 2021
Description: Test the trained KHCNN with rectangular plates
"""

import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_dir, "../"))  # to import other module

from utils import params as PARAMS

from datetime import datetime
from scipy.io import loadmat
import pickle

from utils import data_processing as DATA
from network_module import khcnn_network
from test_module import plot_results
import numpy as np
from utils import metrics as METRICS


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = PARAMS.GPUS
    os.environ['CUDA_ALLOW_GROWTH'] = 'True'

    # select model to use
    if PARAMS.USE_LOW_P_RESOLUTION:
        # hologram pressure with 64 points
        trained_weights_name = "khcnn_rec_64.h5"
    else:
        # hologram pressure with 1024 points
        trained_weights_name = "khcnn_rec_1024.h5"

    now = datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"GPU id:\t\t\t\t{PARAMS.GPUS}")
    print(f"Low P resolution:\t{PARAMS.USE_LOW_P_RESOLUTION}")
    print(f"Trained weights:\t{PARAMS.PATH_TRAINED_WEIGHTS}")
    print("************************************************\n")

    print(f"Load data:\t{PARAMS.PATH_DATA}")

    data = loadmat(PARAMS.PATH_DATA)

    # get data
    freq_test = data["freq"]
    mode_test = data["mode"]
    num_plate_test = data["num_plate"]
    plate_dimensions_test = data["plate_dimensions"]
    boundary_test = data["boundary"]

    # select correct data (hologram pressure at 1024 or 64 points)
    s_size = (16, 64)
    v_s_gt = data["v_s"]
    s_points_test = data["s_points"]

    if PARAMS.USE_LOW_P_RESOLUTION:
        # use downsampling data
        print("P_h points = 64")
        p_h_test = data["p_h_downsampling"]
        h_points_test = data["h_points_downsampling"]
        p_size = (8, 8)
    else:
        print("P_h points = 1024")
        p_h_test = data["p_h"]
        h_points_test = data["h_points"]
        p_size = (16, 64)

    print("Reshape for network...")
    h_points_test = h_points_test.reshape(-1, p_size[0] * p_size[1], 3)
    s_points_test = s_points_test.reshape(-1, s_size[0] * s_size[1], 3)

    print("Normalise wrt abs value...")
    P_test = DATA.normalise_wrt_abs(p_h_test)
    V_test = DATA.normalise_wrt_abs(v_s_gt)

    print("Split real & imag in 2 channels...")
    P_test_ch = DATA.split_real_imag_channels(P_test)
    V_test_ch = DATA.split_real_imag_channels(V_test)

    # compute binary mask
    bm_test = np.ones(shape=V_test_ch.shape)

    # build model
    print("Build model...")
    model, _, _ = khcnn_network.build_khcnn_model(p_size, s_size)

    # get weights
    print("Load weights...")
    model.load_weights(os.path.join(PARAMS.PATH_TRAINED_WEIGHTS, trained_weights_name))

    print("Compute network prediction...")
    estimate_real, estimate_imag, v_s_real_hat, v_s_imag_hat = model.predict([P_test_ch, freq_test, h_points_test, s_points_test, bm_test])
    p_h_hat = estimate_real + 1j*estimate_imag
    v_s_hat = v_s_real_hat + 1j*v_s_imag_hat

    print("Plot results") #================================================================================
    i = np.random.randint(len(P_test))
    tested_info = plot_results.print_information(i, mode_test=mode_test, freq_test=freq_test, num_plate_test=num_plate_test,
                                   boundary_test=boundary_test, plate_dimensions_test=plate_dimensions_test)

    print("\tp_h...")
    plot_results.plot_abs_phase(p_h_hat, P_test, "p_h", i, p_size)
    p_h_nmse_db = METRICS.nmse(x_hat=p_h_hat[i], x=P_test[i])
    p_h_ncc = METRICS.ncc(x_hat=p_h_hat[i], x=P_test[i])

    print("\tv_s...")
    plot_results.plot_abs_phase(v_s_hat, V_test, "v_s", i, s_size)
    v_s_nmse_db = METRICS.nmse(x_hat=v_s_hat[i], x=V_test[i])
    v_s_ncc = METRICS.ncc(x_hat=v_s_hat[i], x=V_test[i])

    print("\tMetrics...")
    print(f"Field\tNMSE\tNCC")
    print(f"V_s\t{np.round(v_s_nmse_db, 2)} dB\t{np.round(v_s_ncc*100, 2)} %")
    print(f"P_h\t{np.round(p_h_nmse_db, 2)} dB\t{np.round(p_h_ncc*100, 2)} %")


