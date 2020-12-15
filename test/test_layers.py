import quartz
from quartz import layers
import unittest
import numpy as np
import ipdb
from parameterized import parameterized
import torch
import torch.nn as nn


class TestLayers(unittest.TestCase):
    @parameterized.expand([
        ((1,1,10,10,), 10),
        ((50,1,120,1,), 84),
        ((500,1,84,1,), 10),
    ])
    def test_fc(self, dim_input, dim_output):
        t_max = 2**8
        np.random.seed(seed=35)
        weights = (np.random.rand(dim_output,np.product(dim_input[1:])) - 0.5) / 5
        biases = (np.random.rand(dim_output) - 0.5) / 2 # np.zeros((dim_output)) #

        loihi_model = quartz.Network([
            layers.InputLayer(dims=dim_input[1:]),
            layers.Dense(weights=weights, biases=biases),
            layers.MonitorLayer(),
        ])
        values = np.random.rand(*dim_input)
        weight_acc = loihi_model.layers[1].weight_acc
        quantized_values = (values*t_max).round()/t_max
        quantized_weights = (weight_acc*weights).round()/weight_acc
        quantized_biases = (biases*t_max).round()/t_max

        model = nn.Sequential(
            nn.Linear(in_features=np.product(dim_input[1:]), out_features=dim_output), 
            nn.ReLU()
        )
        model[0].weight = torch.nn.Parameter(torch.tensor(quantized_weights))
        model[0].bias = torch.nn.Parameter(torch.tensor((quantized_biases)))
        model_output = model(torch.tensor(quantized_values.reshape(dim_input[0], -1))).detach().numpy()
        loihi_output = loihi_model(values, t_max)
        
        self.assertEqual(len(loihi_output.flatten()), len(model_output.flatten()))
        combinations = list(zip(loihi_output.flatten(), model_output.flatten()))
        for (out, ideal) in combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=2)


    @parameterized.expand([
        (( 1, 3,8,8), (  5, 3,2,2), 1, 1),
        (( 1, 4,8,8), (  4, 1,3,3), 1, 4), # depthwise
        (( 1, 6,4,4), (  6, 2,3,3), 0, 3), # grouped
        (( 5, 3,8,8), (  5, 3,1,1), 0, 1), # pointwise
        ((50,10,5,5), (120,10,5,5), 0, 1),
    ])
    def test_conv2d(self, input_dims, weight_dims, padding, groups):
        t_max = 2**8
        kernel_size = weight_dims[2:]
        weights = (np.random.rand(*weight_dims)-0.5) / 4
        biases = (np.random.rand(weight_dims[0])-0.5) / 2 # np.zeros((weight_dims[0])) # 

        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims[1:]),
            layers.Conv2D(weights=weights, biases=biases, padding=padding, groups=groups),
            layers.MonitorLayer(),
        ])

        values = np.random.rand(*input_dims) / 2
        quantized_values = (values*t_max).round()/t_max
        weight_acc = loihi_model.layers[1].weight_acc
        quantized_weights = (weight_acc*weights).round()/weight_acc
        quantized_biases = (biases*t_max).round()/t_max

        model = nn.Sequential(
            nn.Conv2d(in_channels=weight_dims[1]*groups, out_channels=weight_dims[0], kernel_size=kernel_size, padding=padding, groups=groups), 
            nn.ReLU()
        )
        model[0].weight = torch.nn.Parameter(torch.tensor(quantized_weights))
        model[0].bias = torch.nn.Parameter(torch.tensor(quantized_biases))
        model_output = model(torch.tensor(quantized_values)).squeeze().detach().numpy()
        loihi_output = loihi_model(values, t_max)
        
        self.assertEqual(len(loihi_output.flatten()), len(model_output.flatten()))
        output_combinations = list(zip(loihi_output.flatten(), model_output.flatten()))
        #print(output_combinations)
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=2)

            
    @parameterized.expand([
        ((1,1,8,8), (3,1,3,3)),
        ((50,3,6,6), (6,3,5,5)),
        ((500,2,4,4), (4,2,3,3)),
    ])
    def test_convpool2d(self, input_dims, weight_dims):
        t_max = 2**8
        conv_kernel_size = weight_dims[2:]
        pooling_kernel_size = [2,2]
        pooling_stride = 2

        np.random.seed(seed=46)
        np.set_printoptions(suppress=True)
        weights = (np.random.rand(*weight_dims)-0.5) / 4 # np.zeros(weight_dims) #
        biases = (np.random.rand(weight_dims[0])-0.5) / 2 # np.zeros(weight_dims[0]) #

        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims[1:]),
            layers.ConvPool2D(weights=weights, biases=biases, pool_kernel_size=pooling_kernel_size),
            layers.MonitorLayer(),
        ])

        values = np.random.rand(*input_dims)
        quantized_values = (values*t_max).round()/t_max
        weight_acc = loihi_model.layers[1].weight_acc
        quantized_weights = (weight_acc*weights).round()/weight_acc
        quantized_biases = (biases*t_max).round()/t_max

        model = nn.Sequential(
            nn.Conv2d(in_channels=weight_dims[1], out_channels=weight_dims[0], kernel_size=conv_kernel_size), nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride), nn.ReLU(),
        )
        model[0].weight = nn.Parameter(torch.tensor(quantized_weights))
        model[0].bias = nn.Parameter(torch.tensor(quantized_biases))
        model_output = model(torch.tensor(quantized_values)).detach().numpy()
        
        loihi_output = loihi_model(values, t_max)
        
        self.assertEqual(len(loihi_output.flatten()), len(model_output.flatten()))
        output_combinations = list(zip(loihi_output.flatten(), model_output.flatten()))
        #print(output_combinations)
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=2)
