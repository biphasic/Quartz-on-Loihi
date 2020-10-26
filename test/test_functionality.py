import quartz
from quartz import layers
import unittest
import numpy as np
import ipdb
from parameterized import parameterized
import torch
import torch.nn as nn


class TestFunctionality(unittest.TestCase):
    @parameterized.expand([
        ((1,1,1,), 15),
    ])
    def test_bias(self, dim_input, dim_output):
        t_max = 2**8
        np.random.seed(seed=47)
        weights = np.ones((dim_output,np.product(dim_input)))
        biases = (np.random.rand(dim_output) - 0.5)

        loihi_model = quartz.Network([
            layers.InputLayer(dims=dim_input),
            layers.Dense(weights=weights, biases=biases),
            layers.MonitorLayer(),
        ])
        values = np.zeros((1, *dim_input))
        quantized_biases = (biases*t_max).round()/t_max

        loihi_output = loihi_model(values, t_max)
        self.assertEqual(len(loihi_output), len(quantized_biases.flatten()))
        combinations = list(zip(loihi_output, np.maximum(quantized_biases.flatten(), 0)))
        #ipdb.set_trace()
        print(combinations)
        for (out, ideal) in combinations:
            if ideal <= 1: self.assertEqual(out, ideal)


    @parameterized.expand([
        ((1,3,1,), 1),
    ])
    def test_inputs(self, dim_input, dim_output):
        t_max = 2**8
        np.random.seed(seed=47)
        weights = np.ones((dim_output,np.product(dim_input)))

        loihi_model = quartz.Network([
            layers.InputLayer(dims=dim_input),
            layers.Dense(weights=weights, biases=None),
            layers.MonitorLayer(),
        ])
        values = np.random.rand(1, *dim_input) / 2
        quantized_values = (values*t_max).round()/t_max
        
        pt_model = nn.Sequential(
            nn.Linear(in_features=np.product(dim_input), out_features=dim_output), 
            nn.ReLU()
        )
        pt_model[0].weight = torch.nn.Parameter(torch.tensor(weights))
        pt_model[0].bias = torch.nn.Parameter(torch.tensor(np.zeros(dim_output)))
        model_output = pt_model(torch.tensor(quantized_values).flatten()).detach().numpy()

        loihi_output = loihi_model(values, t_max)
        self.assertEqual(len(loihi_output), len(model_output.flatten()))
        combinations = list(zip(loihi_output, np.maximum(model_output.flatten(), 0)))
        print(combinations)
        for (out, ideal) in combinations:
            if ideal <= 1: self.assertEqual(out, ideal)
