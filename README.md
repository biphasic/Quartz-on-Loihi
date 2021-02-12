# Quartz

This package allows you to easily convert a pre-trained analog neural network (ANN) to a spiking neural network (SNN) for efficient inference on Loihi. 
Rather than rate coding, we use the timing between two spikes to encode values. 

## Prerequisites
Note that this package is based on **nxSDK 0.9.8**, a proprietary Intel software package available to the Intel Neuromrophic Research Community. For all other publicly available dependencies, use
```bash
pip install -r requirements.txt
```

## Supported Layers
We support convolutional, dense and maxpooling layers which are modelled after pyTorch layers. You can experiment with dummy inputs for [single](01_single_layers.ipynb) 
and [multilayer](02_multilayer.ipynb) architectures. For convolutional layers, we support different kernel sizes, strides, padding, groups and skip connections, which allows to replicate most modern feedforward ANN architectures.

## Classification Examples
We include two examples of image classification for [MNIST](03_mnist.ipynb) and [CIFAR10](04_cifar10.ipynb) using pre-trained ANN models.
If you want to train your own ANN, have a look at the [CNN](./CNN) folder.

## Run tests
to check that everything is working, run the following terminal command from the main directory:
```bash
SLURM=1 python -m pytest -s -p no:warnings
```
