import quartz
from quartz import layers
import unittest
import numpy as np
import ipdb
from parameterized import parameterized
import torch
import torch.nn as nn


class TestMultiLayer(unittest.TestCase):
    @parameterized.expand([
        ((1,1,10,1,), 10, 10),
        ((50,1,1,10,), 84, 10),
    ])
    def test_2fc(self, input_dims, l1_output_dim, l2_output_dim):
        t_max = 2**8
        np.random.seed(seed=48)
        weights1 = (np.random.rand(l1_output_dim, np.product(input_dims[1:])) - 0.5) / 2
        biases1 = (np.random.rand(l1_output_dim) - 0.5) / 2
        weights2 = (np.random.rand(l2_output_dim, l1_output_dim) - 0.5) / 2
        biases2 = (np.random.rand(l2_output_dim) - 0.5) / 2
        
        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims[1:]),
            layers.Dense(weights=weights1, biases=biases1),
            layers.Dense(weights=weights2, biases=biases2),
            layers.MonitorLayer(),
        ])
        
        values = np.random.rand(*input_dims) / 2
        weight_acc = loihi_model.layers[1].weight_acc
        quantized_values = (values*t_max).round()/t_max
        quantized_weights1 = (weight_acc*weights1).round()/weight_acc
        quantized_weights2 = (weight_acc*weights2).round()/weight_acc
        quantized_biases1 = (biases1*t_max).round()/t_max
        quantized_biases2 = (biases2*t_max).round()/t_max

        model = nn.Sequential(
            nn.Linear(in_features=np.product(input_dims[1:]), out_features=l1_output_dim), nn.ReLU(),
            nn.Linear(in_features=l1_output_dim, out_features=l2_output_dim), nn.ReLU(),
        )
        model[0].weight = nn.Parameter(torch.tensor(quantized_weights1))
        model[0].bias = nn.Parameter(torch.tensor(quantized_biases1))
        model[2].weight = nn.Parameter(torch.tensor(quantized_weights2))
        model[2].bias = nn.Parameter(torch.tensor(quantized_biases2))
        model_output = model(torch.tensor(quantized_values).reshape(input_dims[0],-1)).detach().numpy()
        
        loihi_output = loihi_model(values, t_max)
        self.assertEqual(len(loihi_output.flatten()), len(model_output.flatten()))
        combinations = list(zip(loihi_output.flatten(), model_output.flatten()))
        #print(combinations)
        for (out, ideal) in combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=1)#, delta=0.05)


    @parameterized.expand([
        ((1,1,7,7), (6,1,5,5), (100,6,3,3)),
        ((100,1,7,7), (6,1,5,5), (100,6,3,3)),
    ])
    def test_2conv2d(self, input_dims, conv_weight_dims1, conv_weight_dims2):
        t_max = 2**8
        conv_kernel_size1 = conv_weight_dims1[2:]
        conv_kernel_size2 = conv_weight_dims2[2:]
        conv_out_dim1 = conv_weight_dims1[0]
        conv_out_dim2 = conv_weight_dims2[0]
        
        weights1 = (np.random.rand(*conv_weight_dims1)-0.5) / 2
        biases1 = (np.random.rand(conv_out_dim1)-0.5) / 2
        weights2 = (np.random.rand(*conv_weight_dims2)-0.5) / 2
        biases2 = (np.random.rand(conv_out_dim2)-0.5) / 2
        
        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims[1:]),
            layers.Conv2D(weights=weights1, biases=biases1),
            layers.Conv2D(weights=weights2, biases=biases2),
            layers.MonitorLayer(),
        ])
        
        values = np.random.rand(*input_dims) / 2
        quantized_values = (values*t_max).round()/t_max
        weight_acc = loihi_model.layers[1].weight_acc
        quantized_weights1 = (weight_acc*weights1).round()/weight_acc
        quantized_weights2 = (weight_acc*weights2).round()/weight_acc
        quantized_biases1 = (biases1*t_max).round()/t_max
        quantized_biases2 = (biases2*t_max).round()/t_max

        model = nn.Sequential(
            nn.Conv2d(in_channels=conv_weight_dims1[1], out_channels=conv_weight_dims1[0], kernel_size=conv_kernel_size1), nn.ReLU(),
            nn.Conv2d(in_channels=conv_weight_dims2[1], out_channels=conv_weight_dims2[0], kernel_size=conv_kernel_size2), nn.ReLU(),
        )
        model[0].weight = nn.Parameter(torch.tensor(quantized_weights1))
        model[0].bias = nn.Parameter(torch.tensor(quantized_biases1))
        model[2].weight = nn.Parameter(torch.tensor(quantized_weights2))
        model[2].bias = nn.Parameter(torch.tensor(quantized_biases2))
        model_output = model(torch.tensor(quantized_values)).detach().numpy()

        loihi_output = loihi_model(values, t_max)
        self.assertEqual(len(loihi_output.flatten()), len(model_output.flatten()))
        output_combinations = list(zip(loihi_output.flatten(), model_output.flatten()))
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, delta=0.05)
            else: print("reduce weight or input value")


    @parameterized.expand([
        ((1,1,12,12), (3,3), (2,2)),
        ((100,1,12,12), (3,3), (2,2)),
    ])
    def test_2maxpool(self, input_dims, kernel_size1, kernel_size2):
        t_max = 2**8
        stride1 = kernel_size1[0]
        stride2 = kernel_size2[0]

        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims[1:]),
            layers.MaxPool2D(kernel_size=kernel_size1),
            layers.MaxPool2D(kernel_size=kernel_size2),
            layers.MonitorLayer(),
        ])

        values = np.arange(np.product(input_dims)).reshape(*input_dims) / np.product(input_dims)
        quantized_values = (values*t_max).round()/t_max

        model = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size1, stride=stride1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size2, stride=stride2), nn.ReLU(),
        )
        model_output = model(torch.tensor(quantized_values)).detach().numpy()
        loihi_output = loihi_model(values, t_max)
        self.assertEqual(len(loihi_output.flatten()), len(model_output.flatten()))
        output_combinations = list(zip(loihi_output.flatten(), model_output.flatten()))
        # print(output_combinations)
        for (out, ideal) in output_combinations:
            self.assertEqual(out, ideal)


    @parameterized.expand([
        ((1,6,3,3), (100,6,3,3), 10),
        ((100,8,5,5), (120,8,5,5), 84),
    ])
    def test_conv_fc(self, input_dims, conv_weight_dims, fc_out_dim):
        t_max = 2**9
        conv_kernel_size = conv_weight_dims[2:]
        conv_out_dim = conv_weight_dims[0]
        
        weights1 = (np.random.rand(*conv_weight_dims)-0.5) / 4
        biases1 = (np.random.rand(conv_out_dim)-0.5) / 2
        weights2 = (np.random.rand(fc_out_dim, np.product(conv_out_dim)) - 0.5) / 4
        biases2 = (np.random.rand(fc_out_dim)-0.5) / 2
        
        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims[1:]),
            layers.Conv2D(weights=weights1, biases=biases1),
            layers.Dense(weights=weights2, biases=biases2),
            layers.MonitorLayer(),
        ])

        values = np.random.rand(*input_dims) / 3
        quantized_values = (values*t_max).round()/t_max
        weight_acc = loihi_model.layers[1].weight_acc
        quantized_weights1 = (weight_acc*weights1).round()/weight_acc
        quantized_weights2 = (weight_acc*weights2).round()/weight_acc
        quantized_biases1 = (biases1*t_max).round()/t_max
        quantized_biases2 = (biases2*t_max).round()/t_max

        model = nn.Sequential(
            nn.Conv2d(in_channels=conv_weight_dims[1], out_channels=conv_weight_dims[0], kernel_size=conv_kernel_size), nn.ReLU(),
            nn.Flatten(), nn.Linear(in_features=np.product(conv_weight_dims[:2]), out_features=fc_out_dim), nn.ReLU(),
        )
        model[0].weight = nn.Parameter(torch.tensor(quantized_weights1))
        model[0].bias = nn.Parameter(torch.tensor(quantized_biases1))
        model[3].weight = nn.Parameter(torch.tensor(quantized_weights2))
        model[3].bias = nn.Parameter(torch.tensor(quantized_biases2))
        model_output = model(torch.tensor(quantized_values)).detach().numpy()

        loihi_output = loihi_model(values, t_max)
        self.assertEqual(len(loihi_output.flatten()), len(model_output.flatten()))
        output_combinations = list(zip(loihi_output.flatten(), model_output.flatten()))
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, delta=0.05)


    @parameterized.expand([
        ((1,1,10,10), (6,1,3,3)),
        ((10,1,28,28), (4,1,5,5)),
        ((100,4,14,14), (6,4,5,5)),
    ])
    def test_conv_maxpool_2d(self, input_dims, weight_dims):
        t_max = 2**9
        conv_kernel_size = weight_dims[2:]
        pooling_kernel_size = [2,2]
        pooling_stride = 2

        weights = (np.random.rand(*weight_dims)-0.5) / 4
        biases = (np.random.rand(weight_dims[0])-0.5) / 2

        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims[1:]),
            layers.Conv2D(weights=weights, biases=biases),
            layers.MaxPool2D(kernel_size=pooling_kernel_size, stride=pooling_stride),
            layers.MonitorLayer(),
        ])
        
        values = np.random.rand(*input_dims) / 2
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
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=2)


    @parameterized.expand([
        ((1,1,10,10), (6,1,5,5)),
        ((50,3,24,24), (8,3,5,5)),
    ])
    def test_maxpool_conv(self, input_dims, weight_dims):
        t_max = 2**9
        conv_kernel_size = weight_dims[2:]
        pooling_kernel_size = [2,2]
        pooling_stride = 2
        weights = (np.random.rand(*weight_dims)-0.5) / 4
        biases = (np.random.rand(weight_dims[0])-0.5) / 2

        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims[1:]),
            layers.MaxPool2D(kernel_size=pooling_kernel_size, stride=pooling_stride),
            layers.Conv2D(weights=weights, biases=biases),
            layers.MonitorLayer(),
        ])
        values = np.random.rand(*input_dims) / 2
        quantized_values = (values*t_max).round()/t_max
        weight_acc = loihi_model.layers[1].weight_acc
        quantized_weights = (weight_acc*weights).round()/weight_acc
        quantized_biases = (biases*t_max).round()/t_max

        model = nn.Sequential(
            nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride), nn.ReLU(),
            nn.Conv2d(in_channels=weight_dims[1], out_channels=weight_dims[0], kernel_size=conv_kernel_size), nn.ReLU(),
        )
        model[2].weight = nn.Parameter(torch.tensor(quantized_weights))
        model[2].bias = nn.Parameter(torch.tensor(quantized_biases))
        model_output = model(torch.tensor(quantized_values)).squeeze().detach().numpy()
        
        loihi_output = loihi_model(values, t_max)
        self.assertEqual(len(loihi_output.flatten()), len(model_output.flatten()))
        output_combinations = list(zip(loihi_output.flatten(), model_output.flatten()))
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=2)


    @parameterized.expand([
        ((1,1,14,14), (3,1,3,3), (5,3,3,3)),
        ((100,1,14,14), (3,1,3,3), (5,3,3,3)),
    ])
    def test_2convpool(self, input_dims, weight_dims, weight_dims2):
        t_max = 2**8
        conv_kernel_size = weight_dims[2:]
        pooling_kernel_size = [2,2]
        pooling_stride = 2
        np.random.seed(seed=44)
        np.set_printoptions(suppress=True)
        weights1 = (np.random.rand(*weight_dims)-0.5) / 5 # np.zeros(weight_dims) #
        weights2 = (np.random.rand(*weight_dims2)-0.5) / 5 # np.zeros(weight_dims) #
        biases1 = (np.random.rand(weight_dims[0])-0.5) / 3
        biases2 = (np.random.rand(weight_dims2[0])-0.5) / 3 

        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims[1:]),
            layers.ConvPool2D(weights=weights1, biases=biases1, pool_kernel_size=pooling_kernel_size),
            layers.ConvPool2D(weights=weights2, biases=biases2, pool_kernel_size=pooling_kernel_size),
            layers.MonitorLayer(),
        ])
        loihi_model1 = quartz.Network([
            layers.InputLayer(dims=input_dims[1:]),
            layers.Conv2D(weights=weights1, biases=biases1),
            layers.MaxPool2D(kernel_size=pooling_kernel_size, stride=pooling_stride),
            layers.Conv2D(weights=weights2, biases=biases2),
            layers.MaxPool2D(kernel_size=pooling_kernel_size, stride=pooling_stride),
            layers.MonitorLayer(),
        ])
        values = np.random.rand(*input_dims) / 3
        quantized_values = (values*t_max).round()/t_max
        weight_acc = loihi_model.layers[1].weight_acc
        quantized_weights1 = (weight_acc*weights1).round()/weight_acc
        quantized_weights2 = (weight_acc*weights2).round()/weight_acc
        quantized_biases1 = (biases1*t_max).round()/t_max
        quantized_biases2 = (biases2*t_max).round()/t_max
        
        model = nn.Sequential(
            nn.Conv2d(in_channels=weight_dims[1], out_channels=weight_dims[0], kernel_size=conv_kernel_size), nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride), nn.ReLU(),
            nn.Conv2d(in_channels=weight_dims2[1], out_channels=weight_dims2[0], kernel_size=conv_kernel_size), nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride), nn.ReLU(),
        )
        model[0].weight = nn.Parameter(torch.tensor(quantized_weights1))
        model[0].bias = nn.Parameter(torch.tensor(quantized_biases1))
        model[4].weight = nn.Parameter(torch.tensor(quantized_weights2))
        model[4].bias = nn.Parameter(torch.tensor(quantized_biases2))
        model_output = model(torch.tensor(quantized_values)).detach().numpy()
        loihi_output = loihi_model(values, t_max)
        loihi_output1 = loihi_model1(values, t_max)
        
        self.assertEqual(len(loihi_output.flatten()), len(model_output.flatten()))
        self.assertEqual(len(loihi_output.flatten()), len(loihi_output1.flatten()))
        output_combinations = list(zip(loihi_output.flatten(), model_output.flatten()))
        #print(output_combinations)
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=2)
        output_combinations = list(zip(loihi_output.flatten(), loihi_output1.flatten()))
        for (out, ideal) in output_combinations:
            self.assertEqual(out, ideal)


    @parameterized.expand([
        ((1,1,16,16), (2,1,3,3), (4,2,3,3)),
        ((100,1,16,16), (2,1,3,3), (4,2,3,3)),
    ])
    def test_convpool_conv(self, input_dims, weight_dims, weight_dims2):
        t_max = 2**8
        conv_kernel_size = weight_dims[2:]
        pooling_kernel_size = [2,2]
        pooling_stride = 2

        np.random.seed(seed=44)
        np.set_printoptions(suppress=True)
        weights = (np.random.rand(*weight_dims)-0.5) / 2 # np.zeros(weight_dims) #
        weights2 = (np.random.rand(*weight_dims2)-0.5) / 2 # np.zeros(weight_dims) #
        biases = (np.random.rand(weight_dims[0])-0.5)
        biases2 = (np.random.rand(weight_dims2[0])-0.5)

        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims[1:]),
            layers.ConvPool2D(weights=weights, biases=biases, pool_kernel_size=pooling_kernel_size),
            layers.Conv2D(weights=weights2, biases=biases2),
            layers.MonitorLayer(),
        ])

        values = np.random.rand(*input_dims) / 2
        #weight_acc = loihi_model.layers[1].weight_acc

        model = nn.Sequential(
            nn.Conv2d(in_channels=weight_dims[1], out_channels=weight_dims[0], kernel_size=conv_kernel_size), nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride), nn.ReLU(),
            nn.Conv2d(in_channels=weight_dims2[1], out_channels=weight_dims2[0], kernel_size=conv_kernel_size), nn.ReLU(),
        )
        model[0].weight = nn.Parameter(torch.tensor(weights))
        model[0].bias = nn.Parameter(torch.tensor(biases))
        model[4].weight = nn.Parameter(torch.tensor(weights2))
        model[4].bias = nn.Parameter(torch.tensor(biases2))
        model_output = model(torch.tensor(values)).squeeze().detach().numpy()
        loihi_output = loihi_model(values, t_max)
        
        self.assertEqual(len(loihi_output.flatten()), len(model_output.flatten()))
        output_combinations = list(zip(loihi_output.flatten(), model_output.flatten()))
        # print(output_combinations)
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=2)

