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
        ((10,1,), 10, 10),
        ((120,1,), 84, 10),
    ])
    def test_2fc(self, dim_input, l1_output_dim, l2_output_dim):
        t_max = 2**9
        run_time = 6*t_max
        dims = (1,*dim_input,2)
        weight_e = 400
        weight_acc = 128
        weight_args = {'weight_e':weight_e, 'weight_acc':weight_acc}

        l0 = quartz.layers.InputLayer(dims=dims, monitor=False, **weight_args)
        weights1 = (np.random.rand(l1_output_dim, np.product(dim_input)) - 0.5) / 2
        biases1 = (np.random.rand(l1_output_dim) - 0.5) / 2
        l1 = quartz.layers.FullyConnected(prev_layer=l0, weights=weights1, biases=biases1, split_output=True, **weight_args)
        weights2 = (np.random.rand(l2_output_dim, l1_output_dim) - 0.5) / 2
        biases2 = (np.random.rand(l2_output_dim) - 0.5) / 2
        l2 = quartz.layers.FullyConnected(prev_layer=l1, weights=weights2, biases=biases2, split_output=False, **weight_args)
        l3 = quartz.layers.MonitorLayer(prev_layer=l2, **weight_args)

        values = np.random.rand(np.product(dims)) / 2
        inputs = quartz.utils.decode_values_into_spike_input(values, t_max)
        quantized_values = (values*t_max).round()/t_max
        quantized_weights1 = (weight_acc*weights1).round()/weight_acc
        quantized_weights2 = (weight_acc*weights2).round()/weight_acc
        quantized_biases1 = (biases1*t_max).round()/t_max
        quantized_biases2 = (biases2*t_max).round()/t_max

        model = nn.Sequential(
            nn.Linear(in_features=np.product(dim_input), out_features=l1_output_dim), nn.ReLU(),
            nn.Linear(in_features=l1_output_dim, out_features=l2_output_dim), nn.ReLU(),
        )
        model[0].weight = nn.Parameter(torch.tensor(quantized_weights1))
        model[0].bias = nn.Parameter(torch.tensor(quantized_biases1))
        model[2].weight = nn.Parameter(torch.tensor(quantized_weights2))
        model[2].bias = nn.Parameter(torch.tensor(quantized_biases2))
        model_output = model(torch.tensor(quantized_values)).detach().numpy()
        
        output_values, spike_times = l3.run_on_loihi(run_time, t_max=t_max, input_spike_list=inputs, plot=False)
        loihi_output = [value[0] for (key, value) in sorted(output_values.items())]
        self.assertEqual(len(loihi_output), len(model_output.flatten()))
        combinations = list(zip(loihi_output, model_output.flatten()))
        #print(combinations)
        for (out, ideal) in combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=1)


    @parameterized.expand([
        ((1,7,7,2,), (6,1,5,5), (100,6,3,3), 120),
    ])
    def test_2conv(self, input_dims, conv_weight_dims1, conv_weight_dims2, weight_e):
        t_max = 2**8
        run_time = 8*t_max
        weight_acc = 128
        conv_kernel_size1 = conv_weight_dims1[2:]
        conv_kernel_size2 = conv_weight_dims2[2:]
        conv_out_dim1 = conv_weight_dims1[0]
        conv_out_dim2 = conv_weight_dims2[0]

        l0 = quartz.layers.InputLayer(dims=input_dims, weight_e=weight_e, weight_acc=weight_acc)
        weights1 = (np.random.rand(*conv_weight_dims1)-0.5) / 2
        biases1 = (np.random.rand(conv_out_dim1)-0.5) / 2
        l1 = quartz.layers.Conv2D(prev_layer=l0, weights=weights1, biases=biases1, split_output=True,
                                 monitor=False, weight_e=weight_e, weight_acc=weight_acc)
        weights2 = (np.random.rand(*conv_weight_dims2)-0.5) / 2
        biases2 = (np.random.rand(conv_out_dim2)-0.5) / 2
        l2 = quartz.layers.Conv2D(prev_layer=l1, weights=weights2, biases=biases2, split_output=False,
                                 monitor=False, weight_e=weight_e, weight_acc=weight_acc)
        l3 = quartz.layers.MonitorLayer(prev_layer=l2, weight_e=weight_e, weight_acc=weight_acc)

        values = np.random.rand(np.product(input_dims)) / 2
        inputs = quartz.utils.decode_values_into_spike_input(values, t_max)
        quantized_values = (values*t_max).round()/t_max
        quantized_values = quantized_values.reshape(*input_dims[:3])
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
        model_output = model(torch.tensor(quantized_values.reshape(1, *input_dims[:3]))).squeeze().detach().numpy()

        output_values, spike_times = l3.run_on_loihi(run_time, t_max=t_max, input_spike_list=inputs, plot=False)
        self.assertEqual(len(output_values.items()), len(model_output.flatten()))
        output_combinations = list(zip([value[0] for (key, value) in sorted(output_values.items())], model_output.flatten()))
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=1)
            else: print("reduce weight or input value")


    @parameterized.expand([
        ((1,12,12,2,), (3,3), (2,2)),
    ])
    def test_2maxpool(self, input_dims, kernel_size1, kernel_size2):
        t_max = 2**8
        run_time = 8*t_max
        stride1 = kernel_size1[0]
        stride2 = kernel_size2[0]
        weight_e = 120
        weight_acc = 128
        
        l0 = quartz.layers.InputLayer(dims=input_dims, monitor=False)
        l1 = quartz.layers.MaxPool2D(prev_layer=l0, kernel_size=kernel_size1, split_output=True,
                                    weight_e=weight_e, weight_acc=weight_acc)
        l2 = quartz.layers.MaxPool2D(prev_layer=l1, kernel_size=kernel_size2, split_output=False,
                                    weight_e=weight_e, weight_acc=weight_acc)
        l3 = quartz.layers.MonitorLayer(prev_layer=l2, weight_e=weight_e, weight_acc=weight_acc)

        values = np.arange(np.product(input_dims)) / 200
        inputs = quartz.utils.decode_values_into_spike_input(values, t_max)
        quantized_values = (values*t_max).round()/t_max

        model = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size1, stride=stride1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size2, stride=stride2), nn.ReLU(),
        )
        model_output = model(torch.tensor(quantized_values.reshape(1, *input_dims[:3]))).squeeze().detach().numpy()

        output_values, spike_times = l3.run_on_loihi(run_time, t_max=t_max, input_spike_list=inputs, plot=False)
        self.assertEqual(len(output_values.items()), len(model_output.flatten()))
        output_combinations = list(zip([value[0] for (key, value) in sorted(output_values.items())], model_output.flatten()))
        print(output_combinations)
        for (out, ideal) in output_combinations:
            self.assertEqual(out, ideal)


    @parameterized.expand([
        ((6, 3, 3,2,), (100,6,3,3), 10, 500),
        ((16, 5, 5,2,), (120,16,5,5), 84, 500),
    ])
    def test_conv_fc(self, input_dims, conv_weight_dims, fc_out_dim, weight_e):
        t_max = 2**8
        run_time = 8*t_max
        weight_acc = 128
        conv_kernel_size = conv_weight_dims[2:]
        conv_out_dim = conv_weight_dims[0]

        l0 = quartz.layers.InputLayer(dims=input_dims, weight_e=weight_e, weight_acc=weight_acc)
        weights1 = (np.random.rand(*conv_weight_dims)-0.5) / 4
        biases1 = (np.random.rand(conv_out_dim)-0.5) / 2
        l1 = quartz.layers.Conv2D(prev_layer=l0, weights=weights1, biases=biases1, split_output=True,
                                 monitor=False, weight_e=weight_e, weight_acc=weight_acc)
        weights2 = (np.random.rand(fc_out_dim, np.product(conv_out_dim)) - 0.5) / 2
        biases2 = (np.random.rand(fc_out_dim)-0.5) / 2
        l2 = quartz.layers.FullyConnected(prev_layer=l1, weights=weights2, biases=biases2, split_output=False,
                                         monitor=False, weight_e=weight_e, weight_acc=weight_acc)
        l3 = quartz.layers.MonitorLayer(prev_layer=l2, weight_e=weight_e, weight_acc=weight_acc)

        values = np.random.rand(np.product(input_dims)) / 2
        inputs = quartz.utils.decode_values_into_spike_input(values, t_max)
        quantized_values = (values*t_max).round()/t_max
        quantized_values = quantized_values.reshape(*input_dims[:3])
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
        model_output = model(torch.tensor(quantized_values.reshape(1, *input_dims[:3]))).squeeze().detach().numpy()

        output_values, spike_times = l3.run_on_loihi(run_time, t_max=t_max, input_spike_list=inputs, plot=False)
        self.assertEqual(len(output_values.items()), len(model_output.flatten()))
        output_combinations = list(zip([value[0] for (key, value) in sorted(output_values.items())], model_output.flatten()))
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=1)


    @parameterized.expand([
        (( 1,10,10), ( 6,1,3,3), 250),
        #(( 1,32,32), ( 6,1,5,5), 250),
        #(( 6,14,14), (16,6,5,5), 250),
    ])
    def test_conv_maxpool_2d(self, input_dims, weight_dims, weight_e):
        t_max = 2**9
        weight_acc = 128
        model_args = {'weight_e':weight_e, 'weight_acc':weight_acc}
        conv_kernel_size = weight_dims[2:]
        pooling_kernel_size = [2,2]
        pooling_stride = 2

        weights = (np.random.rand(*weight_dims)-0.5) / 4
        biases = (np.random.rand(weight_dims[0])-0.5) / 2

        loihi_model = quartz.Network([
            layers.InputLayer(dims=input_dims, **model_args),
            layers.Conv2D(weights=weights, biases=biases, **model_args),
            layers.MaxPool2D(kernel_size=pooling_kernel_size, stride=pooling_stride, **model_args),
            layers.MonitorLayer(**model_args),
        ])

        values = np.random.rand(np.product(input_dims))
        inputs = quartz.utils.decode_values_into_spike_input(values, t_max)
        quantized_values = (values*t_max).round()/t_max
        quantized_weights = (weight_acc*weights).round()/weight_acc
        quantized_biases = (biases*t_max).round()/t_max

        model = nn.Sequential(
            nn.Conv2d(in_channels=weight_dims[1], out_channels=weight_dims[0], kernel_size=conv_kernel_size), nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride), nn.ReLU(),
        )
        model[0].weight = nn.Parameter(torch.tensor(quantized_weights))
        model[0].bias = nn.Parameter(torch.tensor(quantized_biases))
        model_output = model(torch.tensor(quantized_values.reshape(1, *input_dims[:3]))).squeeze().detach().numpy()

        output_values = loihi_model(inputs, t_max)
        ipdb.set_trace()
        self.assertEqual(len(output_values), len(model_output.flatten()))
        output_combinations = list(zip([value[0] for (key, value) in sorted(output_values.items())], model_output.flatten()))
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=1)


    @parameterized.expand([
        (( 1,10,10,2), ( 6,1,5,5), 500),
        (( 6,28,28,2), (16,6,5,5), 500),
    ])
    def test_maxpool_conv(self, input_dims, weight_dims, weight_e):
        t_max = 2**9
        run_time = 10*t_max
        weight_acc = 128
        conv_kernel_size = weight_dims[2:]
        pooling_kernel_size = [2,2]
        pooling_stride = 2

        l0 = quartz.layers.InputLayer(dims=input_dims, monitor=False, weight_e=weight_e, weight_acc=weight_acc)
        weights = (np.random.rand(*weight_dims)-0.5) / 5
        biases = (np.random.rand(weight_dims[0])-0.5) / 3
        l1 = quartz.layers.MaxPool2D(prev_layer=l0, kernel_size=pooling_kernel_size, split_output=True,
                                    weight_e=weight_e, weight_acc=weight_acc)
        l2 = quartz.layers.Conv2D(prev_layer=l1, weights=weights, biases=biases, split_output=False,
                                 weight_e=weight_e, weight_acc=weight_acc)
        l3 = quartz.layers.MonitorLayer(prev_layer=l2, weight_e=weight_e, weight_acc=weight_acc)

        values = np.random.rand(np.product(input_dims))
        inputs = quartz.utils.decode_values_into_spike_input(values, t_max)
        quantized_values = (values*t_max).round()/t_max
        quantized_weights = (weight_acc*weights).round()/weight_acc
        quantized_biases = (biases*t_max).round()/t_max

        model = nn.Sequential(
            nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride), nn.ReLU(),
            nn.Conv2d(in_channels=weight_dims[1], out_channels=weight_dims[0], kernel_size=conv_kernel_size), nn.ReLU(),
        )
        model[2].weight = nn.Parameter(torch.tensor(quantized_weights))
        model[2].bias = nn.Parameter(torch.tensor(quantized_biases))
        model_output = model(torch.tensor(quantized_values.reshape(1, *input_dims[:3]))).squeeze().detach().numpy()
        
        output_values, spike_times = l3.run_on_loihi(run_time, t_max=t_max, input_spike_list=inputs, plot=False)
        self.assertEqual(len(output_values.items()), len(model_output.flatten()))
        output_combinations = list(zip([value[0] for (key, value) in sorted(output_values.items())], model_output.flatten()))
        for (out, ideal) in output_combinations:
            if ideal <= 1: self.assertAlmostEqual(out, ideal, places=1)
