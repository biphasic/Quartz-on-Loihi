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
            layers.Dense(weights=weights, biases=biases, split_output=False),
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
        run_time = 4*t_max
        weight_e = 500
        weight_acc = 2**8
        weight_acc_real = weight_acc / 2
        model_args = {'weight_e':weight_e, 'weight_acc':weight_acc}

        np.random.seed(seed=47)
        weights = (np.random.rand(dim_output,np.product(dim_input)) - 0.5) / 5
        biases = (np.random.rand(dim_output) - 0.5) / 3

        loihi_model = quartz.Network([
            layers.InputLayer(dims=dim_input, **model_args),
            layers.Dense(weights=weights, biases=biases, **model_args),
            layers.MonitorLayer(**model_args),
        ])

        values = np.random.rand(np.product(dim_input))
        inputs = quartz.decode_values_into_spike_input(values, t_max)

        quantized_values = (values*t_max).round()/t_max
        quantized_weights = (weight_acc_real*weights).round()/weight_acc_real
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
        (( 3, 8, 8), (  5, 3,2,2), 251),
        ((10, 5, 5), (120,10,5,5), 251),
    ])
    def test_conv2d(self, input_dims, weight_dims, weight_e):
        t_max = 2**9
        weight_acc = 2**7
        model_args = {'weight_e':weight_e, 'weight_acc':weight_acc}

        kernel_size = weight_dims[2:]
        weights = (np.random.rand(*weight_dims)-0.5) / 5
        biases = (np.random.rand(weight_dims[0])-0.5) / 2

        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims, **model_args),
            layers.Conv2D(weights=weights, biases=biases, **model_args),
            layers.MonitorLayer(**model_args),
        ])

        input0 = quartz.probe(loihi_model.layers[0].blocks[0])
        hidden0 = quartz.probe(loihi_model.layers[1].blocks[2])
        hidden1 = quartz.probe(loihi_model.layers[1].blocks[-1])
        calc_probe = quartz.probe(loihi_model.layers[1].blocks[-1].neurons[0])
        sync_probe = quartz.probe(loihi_model.layers[1].blocks[-1].neurons[2])

        values = np.random.rand(np.product(input_dims)) / 2
        inputs = quartz.utils.decode_values_into_spike_input(values, t_max)

        quantized_values = (values*t_max).round()/t_max
        quantized_values = quantized_values.reshape(*input_dims)
        if weight_acc > 255: weight_acc /= 2
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
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=1)


    @parameterized.expand([
        ((1,10,10),),
        ((6,28,28),),
        ((16,10,10),),
    ])
    def test_maxpool2d(self, input_dims):
        t_max = 2**8
        input_dims = (1,10,10)
        kernel_size = [2,2]
        model_args = {'weight_e':200, 'weight_acc':128}

        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims, **model_args),
            layers.MaxPool2D(kernel_size=kernel_size, **model_args),
            layers.MonitorLayer(**model_args),
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
