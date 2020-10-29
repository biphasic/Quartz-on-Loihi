import quartz
from quartz import layers
import unittest
import numpy as np
import ipdb
from parameterized import parameterized
from sklearn.feature_extraction import image
import torch
import torch.nn as nn
import collections


class TestMultiLayer(unittest.TestCase):
    @parameterized.expand([
        #((1,10,1,), 10, 10),
        ((1,120,1,), 84, 10),
    ])
    def test_2fc(self, input_dims, l1_output_dim, l2_output_dim):
        t_max = 2**9
        batch_size = 1
        np.random.seed(seed=48)
        weights1 = (np.random.rand(l1_output_dim, np.product(input_dims)) - 0.5) / 2
        biases1 = (np.random.rand(l1_output_dim) - 0.5) / 2
        weights2 = (np.random.rand(l2_output_dim, l1_output_dim) - 0.5) / 2
        biases2 = (np.random.rand(l2_output_dim) - 0.5) / 2
        
        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims),
            layers.Dense(weights=weights1, biases=biases1),
            layers.Dense(weights=weights2, biases=biases2),
            layers.MonitorLayer(),
        ])
        
        values = np.random.rand(batch_size, np.product(input_dims)) / 2
        weight_acc = loihi_model.layers[1].weight_acc
        quantized_values = (values*t_max).round()/t_max
        quantized_weights1 = (weight_acc*weights1).round()/weight_acc
        quantized_weights2 = (weight_acc*weights2).round()/weight_acc
        quantized_biases1 = (biases1*t_max).round()/t_max
        quantized_biases2 = (biases2*t_max).round()/t_max

        model = nn.Sequential(
            nn.Linear(in_features=np.product(input_dims), out_features=l1_output_dim), nn.ReLU(),
            nn.Linear(in_features=l1_output_dim, out_features=l2_output_dim), nn.ReLU(),
        )
        model[0].weight = nn.Parameter(torch.tensor(quantized_weights1))
        model[0].bias = nn.Parameter(torch.tensor(quantized_biases1))
        model[2].weight = nn.Parameter(torch.tensor(quantized_weights2))
        model[2].bias = nn.Parameter(torch.tensor(quantized_biases2))
        model_output = model(torch.tensor(quantized_values)).detach().numpy()
        
        loihi_output = loihi_model(values, t_max)
        self.assertEqual(len(loihi_output), len(model_output.flatten()))
        combinations = list(zip(loihi_output, model_output.flatten()))
        print(combinations)
        for (out, ideal) in combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=1)


    @parameterized.expand([
        ((1,7,7), (6,1,5,5), (100,6,3,3)),
    ])
    def test_2conv(self, input_dims, conv_weight_dims1, conv_weight_dims2):
        t_max = 2**8
        batch_size = 1
        conv_kernel_size1 = conv_weight_dims1[2:]
        conv_kernel_size2 = conv_weight_dims2[2:]
        conv_out_dim1 = conv_weight_dims1[0]
        conv_out_dim2 = conv_weight_dims2[0]
        
        weights1 = (np.random.rand(*conv_weight_dims1)-0.5) / 2
        biases1 = (np.random.rand(conv_out_dim1)-0.5) / 2
        weights2 = (np.random.rand(*conv_weight_dims2)-0.5) / 2
        biases2 = (np.random.rand(conv_out_dim2)-0.5) / 2
        
        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims),
            layers.Conv2D(weights=weights1, biases=biases1),
            layers.Conv2D(weights=weights2, biases=biases2),
            layers.MonitorLayer(),
        ])
        
        values = np.random.rand(batch_size, *input_dims) / 2
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
        self.assertEqual(len(loihi_output), len(model_output.flatten()))
        output_combinations = list(zip(loihi_output, model_output.flatten()))
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=1)
            else: print("reduce weight or input value")


    @parameterized.expand([
        ((1,12,12), (3,3), (2,2)),
    ])
    def test_2maxpool(self, input_dims, kernel_size1, kernel_size2):
        t_max = 2**8
        batch_size = 1
        stride1 = kernel_size1[0]
        stride2 = kernel_size2[0]

        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims),
            layers.MaxPool2D(kernel_size=kernel_size1),
            layers.MaxPool2D(kernel_size=kernel_size2),
            layers.MonitorLayer(),
        ])

        values = np.arange(batch_size * np.product(input_dims)).reshape(batch_size, *input_dims) / 200
        quantized_values = (values*t_max).round()/t_max

        model = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size1, stride=stride1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size2, stride=stride2), nn.ReLU(),
        )
        model_output = model(torch.tensor(quantized_values)).detach().numpy()
        loihi_output = loihi_model(values, t_max)
        self.assertEqual(len(loihi_output), len(model_output.flatten()))
        output_combinations = list(zip(loihi_output, model_output.flatten()))
        # print(output_combinations)
        for (out, ideal) in output_combinations:
            self.assertEqual(out, ideal)


    @parameterized.expand([
        ((6, 3, 3), (100,6,3,3), 10),
        ((8, 5, 5), (120,8,5,5), 84),
    ])
    def test_conv_fc(self, input_dims, conv_weight_dims, fc_out_dim):
        t_max = 2**8
        batch_size = 1
        conv_kernel_size = conv_weight_dims[2:]
        conv_out_dim = conv_weight_dims[0]
        
        weights1 = (np.random.rand(*conv_weight_dims)-0.5) / 4
        biases1 = (np.random.rand(conv_out_dim)-0.5) / 2
        weights2 = (np.random.rand(fc_out_dim, np.product(conv_out_dim)) - 0.5) / 2
        biases2 = (np.random.rand(fc_out_dim)-0.5) / 2
        
        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims),
            layers.Conv2D(weights=weights1, biases=biases1),
            layers.Dense(weights=weights2, biases=biases2),
            layers.MonitorLayer(),
        ])

        values = np.random.rand(batch_size, *input_dims) / 2
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
        self.assertEqual(len(loihi_output), len(model_output.flatten()))
        output_combinations = list(zip(loihi_output, model_output.flatten()))
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=2)


    @parameterized.expand([
        ((1,10,10), (6,1,3,3)),
        ((1,32,32), (6,1,5,5)),
        ((4,14,14), (6,4,5,5)),
    ])
    def test_conv_maxpool_2d(self, input_dims, weight_dims):
        t_max = 2**9
        batch_size = 1
        conv_kernel_size = weight_dims[2:]
        pooling_kernel_size = [2,2]
        pooling_stride = 2

        weights = (np.random.rand(*weight_dims)-0.5) / 4
        biases = (np.random.rand(weight_dims[0])-0.5) / 2

        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims),
            layers.Conv2D(weights=weights, biases=biases),
            layers.MaxPool2D(kernel_size=pooling_kernel_size, stride=pooling_stride),
            layers.MonitorLayer(),
        ])
        
        values = np.random.rand(batch_size, *input_dims) / 2
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
        model_output = model(torch.tensor(quantized_values)).squeeze().detach().numpy()

        loihi_output = loihi_model(values, t_max)
        self.assertEqual(len(loihi_output), len(model_output.flatten()))
        output_combinations = list(zip(loihi_output, model_output.flatten()))
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=2)


    @parameterized.expand([
        ((1,10,10), (6,1,5,5)),
        ((6,28,28), (8,6,5,5)),
    ])
    def test_maxpool_conv(self, input_dims, weight_dims):
        t_max = 2**9
        batch_size = 1
        conv_kernel_size = weight_dims[2:]
        pooling_kernel_size = [2,2]
        pooling_stride = 2
        
        weights = (np.random.rand(*weight_dims)-0.5) / 4
        biases = (np.random.rand(weight_dims[0])-0.5) / 2

        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims),
            layers.MaxPool2D(kernel_size=pooling_kernel_size, stride=pooling_stride),
            layers.Conv2D(weights=weights, biases=biases),
            layers.MonitorLayer(),
        ])

        values = np.random.rand(batch_size, *input_dims) / 2
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
        self.assertEqual(len(loihi_output), len(model_output.flatten()))
        output_combinations = list(zip(loihi_output, model_output.flatten()))
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=2)


    def test_convpool2(self):
        input_dims = ( 1,16,16)
        weight_dims = ( 3,1,3,3)
        weight_dims2 = (5,3,3,3)
        t_max = 2**9
        batch_size = 1
        conv_kernel_size = weight_dims[2:]
        pooling_kernel_size = [2,2]
        pooling_stride = 2

        np.random.seed(seed=44)
        np.set_printoptions(suppress=True)
        weights = (np.random.rand(*weight_dims)-0.5) / 2 # np.zeros(weight_dims) #
        weights2 = (np.random.rand(*weight_dims2)-0.5) / 2 # np.zeros(weight_dims) #
        #weight_acc = loihi_model.layers[1].weight_acc
        biases = (np.random.rand(weight_dims[0])-0.5)
        biases2 = (np.random.rand(weight_dims2[0])-0.5)

        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims),
            layers.ConvPool2D(weights=weights, biases=biases, pool_kernel_size=pooling_kernel_size),
            layers.ConvPool2D(weights=weights2, biases=biases2, pool_kernel_size=pooling_kernel_size),
            layers.MonitorLayer(),
        ])

        values = np.random.rand(batch_size, *input_dims) / 2

        model = nn.Sequential(
            nn.Conv2d(in_channels=weight_dims[1], out_channels=weight_dims[0], kernel_size=conv_kernel_size), nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride), nn.ReLU(),
            nn.Conv2d(in_channels=weight_dims2[1], out_channels=weight_dims2[0], kernel_size=conv_kernel_size), nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride), nn.ReLU(),
        )
        model[0].weight = nn.Parameter(torch.tensor(weights))
        model[0].bias = nn.Parameter(torch.tensor(biases))
        model[4].weight = nn.Parameter(torch.tensor(weights2))
        model[4].bias = nn.Parameter(torch.tensor(biases2))
        model_output = model(torch.tensor(values)).detach().numpy()
        loihi_output = loihi_model(values, t_max)
        
        self.assertEqual(len(loihi_output), len(model_output.flatten()))
        output_combinations = list(zip(loihi_output, model_output.flatten()))
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=1)

    def test_convpool_conv(self):
        input_dims = ( 1,16,16)
        weight_dims = ( 2,1,3,3)
        weight_dims2 = (4,2,3,3)
        t_max = 2**9
        batch_size = 1
        conv_kernel_size = weight_dims[2:]
        pooling_kernel_size = [2,2]
        pooling_stride = 2

        np.random.seed(seed=44)
        np.set_printoptions(suppress=True)
        weights = (np.random.rand(*weight_dims)-0.5) / 2 # np.zeros(weight_dims) #
        weights2 = (np.random.rand(*weight_dims2)-0.5) / 2 # np.zeros(weight_dims) #
        #weight_acc = loihi_model.layers[1].weight_acc
        biases = (np.random.rand(weight_dims[0])-0.5)
        biases2 = (np.random.rand(weight_dims2[0])-0.5)

        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims),
            layers.ConvPool2D(weights=weights, biases=biases, pool_kernel_size=pooling_kernel_size),
            layers.Conv2D(weights=weights2, biases=biases2),
            layers.MonitorLayer(),
        ])

        values = np.random.rand(batch_size, *input_dims) / 2

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
        
        self.assertEqual(len(loihi_output), len(model_output.flatten()))
        output_combinations = list(zip(loihi_output, model_output.flatten()))
        ipdb.set_trace()
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=1)

