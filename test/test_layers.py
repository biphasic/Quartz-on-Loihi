import quartz
from quartz import layers
import unittest
import numpy as np
import ipdb
from parameterized import parameterized
import torch
import torch.nn as nn


class TestLayers(unittest.TestCase):
    def test_model_fc(self):
        dim_input = 100
        dim_output = 100
        weights = (np.random.rand(dim_output,np.product(dim_input)) - 0.5) / 5
        biases = (np.random.rand(dim_output) - 0.5) / 2

        loihi_model = quartz.Network([
            layers.InputLayer(dims=(1,10,10,2)),
            layers.Dense(weights=weights, biases=biases),
            layers.MonitorLayer(),
        ])
        self.assertEqual(loihi_model.n_compartments(), 1400)
        self.assertEqual(loihi_model.n_connections(), 32100)
        self.assertEqual(loihi_model.n_parameters(), 10100)


    @parameterized.expand([
        ((1,10,10,), 10),
        ((1,120,1,), 84),
        ((1,84,1,), 10),
    ])
    def test_fc(self, dim_input, dim_output):
        t_max = 2**9

        np.random.seed(seed=47)
        weights = (np.random.rand(dim_output,np.product(dim_input)) - 0.5) / 5
        biases = (np.random.rand(dim_output) - 0.5) / 3

        loihi_model = quartz.Network([
            layers.InputLayer(dims=dim_input),
            layers.Dense(weights=weights, biases=biases),
            layers.MonitorLayer(),
        ])

        values = np.random.rand(np.product(dim_input))
        inputs = quartz.decode_values_into_spike_input(values, t_max)

        weight_acc = 128
        quantized_values = (values*t_max).round()/t_max
        quantized_weights = (weight_acc*weights).round()/weight_acc
        quantized_biases = (biases*t_max).round()/t_max

        pt_model = nn.Sequential(
            nn.Linear(in_features=np.product(dim_input), out_features=dim_output), 
            nn.ReLU()
        )
        pt_model[0].weight = torch.nn.Parameter(torch.tensor(quantized_weights))
        pt_model[0].bias = torch.nn.Parameter(torch.tensor((quantized_biases)))
        pt_model_output = pt_model(torch.tensor(quantized_values)).detach().numpy()
        loihi_model_output = loihi_model(inputs, t_max)
        combinations = list(zip(loihi_model_output, pt_model_output.flatten()))
        for (out, ideal) in combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=2)


    @parameterized.expand([
        (( 3, 8, 8), (  5, 3,2,2)),
        ((10, 5, 5), (120,10,5,5)),
    ])
    def test_conv2d(self, input_dims, weight_dims):
        t_max = 2**9

        kernel_size = weight_dims[2:]
        weights = (np.random.rand(*weight_dims)-0.5) / 4
        biases = (np.random.rand(weight_dims[0])-0.5) / 2

        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims),
            layers.Conv2D(weights=weights, biases=biases),
            layers.MonitorLayer(),
        ])

        values = np.random.rand(np.product(input_dims)) / 2
        inputs = quartz.utils.decode_values_into_spike_input(values, t_max)

        quantized_values = (values*t_max).round()/t_max
        quantized_values = quantized_values.reshape(*input_dims)
        weight_acc = 2**7
        quantized_weights = (weight_acc*weights).round()/weight_acc
        quantized_biases = (biases*t_max).round()/t_max

        model = nn.Sequential(nn.Conv2d(in_channels=weight_dims[1], out_channels=weight_dims[0], kernel_size=kernel_size), nn.ReLU())
        model[0].weight = torch.nn.Parameter(torch.tensor(quantized_weights))
        model[0].bias = torch.nn.Parameter(torch.tensor(quantized_biases))
        model_output = model(torch.tensor(values.reshape(1, *input_dims[:3]))).squeeze().detach().numpy()
        output_values = loihi_model(inputs, t_max)
        
        self.assertEqual(len(output_values), len(model_output.flatten()))
        output_combinations = list(zip(output_values, model_output.flatten()))
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=2)


    @parameterized.expand([
        ((1,10,10),),
        ((6,28,28),),
        ((16,10,10),),
    ])
    def test_maxpool2d(self, input_dims):
        t_max = 2**8
        input_dims = (1,10,10)
        kernel_size = [2,2]

        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims),
            layers.MaxPool2D(kernel_size=kernel_size),
            layers.MonitorLayer(),
        ])

        np.random.seed(seed=45)
        values = np.random.rand(np.product(input_dims))
        inputs = quartz.utils.decode_values_into_spike_input(values, t_max)
        quantized_values = (values*t_max).round()/t_max

        model = nn.Sequential(nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size[0]), nn.ReLU())
        model_output = model(torch.tensor(quantized_values.reshape(1, *input_dims[:3]))).squeeze().detach().numpy()
        output_values = loihi_model(inputs, t_max)

        self.assertEqual(len(output_values), len(model_output.flatten()))
        output_combinations = list(zip(output_values, model_output.flatten()))
        for (out, ideal) in output_combinations:
            self.assertEqual(out, ideal)
