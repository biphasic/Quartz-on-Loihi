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
        ((50,1,100,1,), 10),
        ((500,1,84,1,), 10),
    ])
    def test_fc(self, dim_input, dim_output):
        t_max = 2**8
        np.random.seed(seed=35)
        weights = (np.random.rand(dim_output,np.product(dim_input[1:])) - 0.5) / 3
        biases = (np.random.rand(dim_output) - 0.5) / 2 # np.zeros((dim_output)) # 
        inputs = np.random.rand(*dim_input) / 3

        loihi_model = quartz.Network(t_max=t_max, layers=[
            layers.InputLayer(dims=dim_input[1:]),
            layers.Dense(weights=weights, biases=biases),
        ])

        model = nn.Sequential(
            nn.Linear(in_features=np.product(dim_input[1:]), out_features=dim_output), 
            nn.ReLU()
        )
        q_weights, q_biases, q_inputs = quartz.utils.quantize_values(weights, biases, inputs, loihi_model.layers[1].weight_acc, t_max)
        model[0].weight = torch.nn.Parameter(torch.tensor(q_weights))
        model[0].bias = torch.nn.Parameter(torch.tensor((q_biases)))
        model_output = model(torch.tensor(q_inputs.reshape(dim_input[0], -1))).detach().numpy()
        loihi_output = loihi_model(inputs)
        
        self.assertEqual(len(loihi_output.flatten()), len(model_output.flatten()))
        combinations = list(zip(loihi_output.flatten(), model_output.flatten()))
        for (out, ideal) in combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=2)


    @parameterized.expand([
        (( 1, 3,8,8), (5, 3,2,2), 1, 1, 1), # padding
        (( 1, 3,7,7), (5, 3,3,3), 2, 0, 1), # stride
        (( 1, 4,8,8), (4, 1,3,3), 1, 1, 4), # depthwise
        (( 1, 6,4,4), (6, 2,3,3), 1, 0, 3), # grouped
        (( 5, 3,8,8), (5, 3,1,1), 1, 0, 1), # pointwise
        ((50,10,5,5), (5,10,5,5), 1, 0, 1), # batch
    ])
    def test_conv2d(self, input_dims, weight_dims, stride, padding, groups):
        t_max = 2**8
        kernel_size = weight_dims[2:]
        weights = (np.random.rand(*weight_dims)-0.5) / 3
        biases = (np.random.rand(weight_dims[0])-0.5) / 2 # np.zeros((weight_dims[0])) # 
        inputs = np.random.rand(*input_dims) / 3

        loihi_model = quartz.Network(t_max, [
            layers.InputLayer(dims=input_dims[1:]),
            layers.Conv2D(weights=weights, biases=biases, stride=stride, padding=padding, groups=groups),
        ])

        model = nn.Sequential(
            nn.Conv2d(in_channels=weight_dims[1]*groups, out_channels=weight_dims[0], kernel_size=kernel_size, stride=stride, padding=padding, groups=groups), 
            nn.ReLU()
        )
        q_weights, q_biases, q_inputs = quartz.utils.quantize_values(weights, biases, inputs, loihi_model.layers[1].weight_acc, t_max)
        model[0].weight = torch.nn.Parameter(torch.tensor(q_weights))
        model[0].bias = torch.nn.Parameter(torch.tensor(q_biases))
        model_output = model(torch.tensor(q_inputs)).squeeze().detach().numpy()
        loihi_output = loihi_model(inputs)
        
        self.assertEqual(len(loihi_output.flatten()), len(model_output.flatten()))
        output_combinations = list(zip(loihi_output.flatten(), model_output.flatten()))
        #print(output_combinations)
        for (out, ideal) in output_combinations:
            if 0 < ideal < 1: self.assertAlmostEqual(out, ideal, places=2)

            
    @parameterized.expand([
        ((1,1,8,8), (2,2)),
        ((50,3,6,6), (3,3)),
    ])
    def test_maxpool2d(self, input_dims, kernel_size):
        # caution: kernel_size =< 3 when directly after InputLayer, as input generators cannot be deactivated by inhibitory connection from MaxPool layer
        t_max = 2**7
        np.random.seed(seed=27)
        inputs = np.random.rand(*input_dims) / 2

        loihi_model = quartz.Network(t_max, [
            layers.InputLayer(dims=input_dims[1:]),
            layers.MaxPool2D(kernel_size=kernel_size)
        ])
        loihi_output = loihi_model(inputs)

        model = nn.MaxPool2d(kernel_size=kernel_size)
        model_output = model(torch.tensor((inputs*t_max).round()/t_max)).detach().numpy()
        
        self.assertEqual(len(loihi_output.flatten()), len(model_output.flatten()))
        output_combinations = list(zip(loihi_output.flatten(), model_output.flatten()))
        #print(output_combinations)
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertEqual(out, ideal)


    @parameterized.expand([
        ((1,1,8,8), (3,1,3,3)),
        ((50,3,6,6), (6,3,5,5)),
        ((100,2,4,4), (4,2,3,3)),
    ])
    def test_convpool2d(self, input_dims, weight_dims):
        t_max = 2**8
        kernel_size = [2,2]

        np.random.seed(seed=27)
        weights = (np.random.rand(*weight_dims)-0.5) / 4 # 
        biases = np.zeros((weight_dims[0])) 
        inputs = np.random.rand(*input_dims) / 2 # np.ones((input_dims)) / 2 #

        loihi_model = quartz.Network(t_max, [
            layers.InputLayer(dims=input_dims[1:]),
            layers.Conv2D(weights=weights),
            layers.MaxPool2D(kernel_size=kernel_size)
        ])

        q_weights, q_biases, q_inputs = quartz.utils.quantize_values(weights, biases, inputs, loihi_model.layers[1].weight_acc, t_max)
        model = nn.Sequential(
            nn.Conv2d(in_channels=weight_dims[1], out_channels=weight_dims[0], kernel_size=weight_dims[2]), nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size),
        )
        model[0].weight = torch.nn.Parameter(torch.tensor(q_weights))
        model[0].bias = torch.nn.Parameter(torch.tensor((q_biases)))
        model_output = model(torch.tensor(q_inputs)).detach().numpy()
        
        loihi_output = loihi_model(inputs)
        
        self.assertEqual(len(loihi_output.flatten()), len(model_output.flatten()))
        output_combinations = list(zip(loihi_output.flatten(), model_output.flatten()))
        #print(output_combinations)
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=1)
