# A physics-informed neural network approach for Nearfield Acoustic Holography
Repository of ["A physics-informed neural network approach for Nearfield Acoustic Holography"](https://) 

M. Olivieri, M. Pezzoli, F. Antonacci, A. Sarti

MDPI, Sensors, 2021
_Special Issue in "Audio Signal Processing for Sensing Technologies"_

In this manuscript we describe a novel methodology for Nearfield Acoustic Holography (NAH). The proposed technique is based on Convolutional Neural Networks, with an autoencoder archictecure, to reconstruct the pressure and velocity fields on the surface of the vibrating structure using the sampled pressure soundfield on the holographic plane as input. The loss function used for training the network is based on a combination of two components. The first component is the error in the reconstructed velocity. The second component is the error between the sound pressure on the holographic plane and its estimate obtained from forward propagating the pressure and velocity fields on the structure through the Kirchhoff-Helmholtz integral, thus bringing some knowledge about the physics of the process under study into the estimation algorithm. Due to the explicit presence of the Kirchhoff-Helmholtz integral in the loss function, we name the proposed technique as Kirchhoff-Helmholtz-based Convolutional Neural Network, KHCNN. KHCNN has been tested on two large datasets of rectangular plates and violin shells.
Results show that it attains a very good accuracy, with a gain in the NMSE of the estimated velocity field that can top 10 dB with respect to state-of-the-art techniques. The same trend is observed if the Normalized Cross Correlation is used as a metric.

## Proposed architecture

![alt text](https://github.com/polimi-ispl/nah-srcnn/blob/main/images/srcnn_architecture.png)

## About the code
The repo code is structured in the following folders:

* _"src"_ contains the following scipts:
  * _"..."_ contains the proposed architecture.
  * _"example.ipynb"_ contains a complete explanation for using the architecture. In particular you can find the test phase related to the paper reconstruction examples.
* _"data"_ contains the weights to test the trained model, the acoustic pressure measurements and velocity, ground truth and the violin binary masks realted to the reconstruction examples to use in the _example.ipynb_ file. All data are save in pickle format.

All the code file is given in Notebook files in order to provide complete explanations.
Scripts are exportable in Python language and the installation of the following modules is required:
* numpy
* tensorflow
* keras
* sklearn.metrics
* pickle (for reading and writing files such as datasets and saved models)
* matplotlib.pyplot (for data visualization with plots)
