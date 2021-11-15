# A physics-informed neural network approach for Nearfield Acoustic Holography
Repository of ["A physics-informed neural network approach for Nearfield Acoustic Holography"](https://) 

M. Olivieri, M. Pezzoli, F. Antonacci, A. Sarti

MDPI, Sensors, 2021 </br>
_Special Issue in "Audio Signal Processing for Sensing Technologies"_

In this manuscript we describe a novel methodology for Nearfield Acoustic Holography (NAH). The proposed technique is based on Convolutional Neural Networks, with an autoencoder archictecure, to reconstruct the pressure and velocity fields on the surface of the vibrating structure using the sampled pressure soundfield on the holographic plane as input. The loss function used for training the network is based on a combination of two components. The first component is the error in the reconstructed velocity. The second component is the error between the sound pressure on the holographic plane and its estimate obtained from forward propagating the pressure and velocity fields on the structure through the Kirchhoff-Helmholtz integral, thus bringing some knowledge about the physics of the process under study into the estimation algorithm. Due to the explicit presence of the Kirchhoff-Helmholtz integral in the loss function, we name the proposed technique as Kirchhoff-Helmholtz-based Convolutional Neural Network, KHCNN. KHCNN has been tested on two large datasets of rectangular plates and violin shells.
Results show that it attains a very good accuracy, with a gain in the NMSE of the estimated velocity field that can top 10 dB with respect to state-of-the-art techniques. The same trend is observed if the Normalized Cross Correlation is used as a metric.

## Proposed architecture

![alt text](https://github.com/polimi-ispl/nah-khcnn/blob/main/images/KHCNN_architecture.png)
Overall scheme of the proposed KHCNN model. The CNN architecture (yellow block) 􏰉􏰉predicts the real and imaginary parts of **_Ps_** and **_V_** from the input **_Ph_**. The two outputs are then 􏰉propagated with the KH model in order to obtain the estimate of **_Ph_**. A proper loss function is built on top of the velocity ground truth and the pressure at the hologram.

## About the code
The repo code is structured in the following folders:
```
├── src                     # code
│   ├── network_util           # comment
│   │   ├── ...
│   │   └── ...
│   └── example.py             # comment
│       ├── diffuse
│       └── soi
├── ...
├── ...                     
```

All the code file is given in Notebook files in order to provide complete explanations.
Scripts are exportable in Python language and the installation of the following modules is required:
* numpy
* tensorflow
* keras
* sklearn.metrics
* pickle
* matplotlib.pyplot
