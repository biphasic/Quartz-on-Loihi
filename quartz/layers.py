from quartz.components import Neuron, Synapse
import quartz
from sklearn.feature_extraction import image
import numpy as np
import ipdb


class Layer:
    def __init__(self, name, weight_e, weight_acc, t_syn=0, t_min=1, t_neu=1):
        self.name = name
        self.weight_e = weight_e
        self.weight_acc = weight_acc
        self.t_syn = t_syn
        self.t_min = t_min
        self.t_neu = t_neu
        self.output_dims = []
        self.layer_n = None
        self.prev_layer = None
        self.blocks = []
        self.neurons = []

    def _get_neurons_of_type(self, neuron_type):
        return [neuron for neuron in self.neurons if neuron.type == neuron_type]

    def input_neurons(self): return self._get_neurons_of_type(Neuron.input)

    def output_neurons(self): return self._get_neurons_of_type(Neuron.output)

    def get_params_at_once(self):
        return self.weight_e, self.weight_acc, self.t_syn, self.t_min, self.t_neu
    
    def neurons(self):
        return [block.neurons for block in self.blocks]

    def n_compartments(self):
        return sum([block.n_compartments() for block in self.blocks])
    
    def n_connections(self):
        return sum([block.n_connections() for block in self.blocks])

    def check_block_delays(self, block, t_max, numDendriticAccumulators):
        for block in self.blocks:
            if isinstance(block, quartz.blocks.ConstantDelay):
                block.reset()
                block.layout_delays(t_max, numDendriticAccumulators)
            
    def print_connections(self, maximum=10e7):
        for i, block in enumerate(self.blocks):
            block.print_connections()
            if i > maximum: break

    def __repr__(self):
        return self.name


class InputLayer(Layer):
    def __init__(self, dims, name="input:", monitor=False, **kwargs):
        super(InputLayer, self).__init__(name=name, **kwargs)
        self.layer_n = 0
        self.output_dims = dims

        for channel in range(dims[0]):
            for height in range(dims[2]):
                for width in range(dims[1]):
                    splitter = quartz.blocks.Splitter(name=name+"split-c{}w{}h{}-".format(channel,height,width), promoted=True, monitor=monitor, parent_layer=self)
                    self.blocks.append(splitter)
                    self.neurons += splitter.input_neurons()
                    self.neurons += splitter.output_neurons()


class FullyConnected(Layer):
    def __init__(self, weights, biases, name="fc", split_output=True, monitor=False, **kwargs):
        super(FullyConnected, self).__init__(name=name, **kwargs)
        self.weights = weights
        self.biases = biases
        self.name = name
        self.output_dims = weights.shape[0]
        self.split_output = split_output
        self.monitor = monitor

    def connect_from(self, prev_layer):
        self.prev_layer = prev_layer
        self.layer_n = prev_layer.layer_n + 1
        self.name += str(self.layer_n)+":"
        weights, biases = self.weights, self.biases
        input_neurons = prev_layer.output_neurons()
        firsts = list(input_neurons[::2])
        seconds = list(input_neurons[1::2])
        assert weights.shape[1]*2 == len(input_neurons)
        if biases != None: assert weights.shape[0] == biases.shape[0]
        assert len(input_neurons) <= self.weight_e

        for i in range(self.output_dims):
            if biases != None:
                bias = quartz.blocks.ConstantDelay(value=biases[i], monitor=False, name=self.name+"l{0}-b{1}-".format(self.layer_n, i), parent_layer=self)
                splitter = quartz.blocks.Splitter(name=self.name+"l{0}-bias{1}-split-".format(self.layer_n, i), promoted=False, monitor=False, parent_layer=self)
                bias.output_neurons()[0].connect_to(splitter.input_neurons()[0], self.weight_e, self.t_syn)
                firsts[i].connect_to(bias.recall_neurons()[0], self.weight_e, self.t_syn)
                bias_sign = 1 if biases[i] >= 0 else -1
                weight_biased = list(weights[i,:])
                weight_biased.append(bias_sign)
                inputs = list(zip(firsts+[splitter.first()], seconds+[splitter.second()], weight_biased))
                self.blocks += [bias, splitter]
            else:
                inputs = list(zip(firsts, seconds, list(weights[i,:])))
            relco = quartz.blocks.ReLCo(inputs, split_input=True, split_output=self.split_output, monitor=self.monitor,
                                     name=self.name+"l{0:1.0f}-n{1:3.0f}-".format(self.layer_n, i), parent_layer=self)
            self.blocks += [relco]
            self.neurons += relco.output_neurons()


class Conv2D(Layer):
    def __init__(self, weights, biases, stride=1, split_output=True, name="conv2D:", monitor=False, **kwargs):
        super(Conv2D, self).__init__(name=name, **kwargs)
        self.weights = weights
        self.biases = biases
        self.stride = stride
        self.name = name
        self.output_dims = weights.shape[0]
        self.split_output = split_output
        self.monitor = monitor

    def connect_from(self, prev_layer):
        self.prev_layer = prev_layer
        self.layer_n = prev_layer.layer_n + 1
        self.name += str(self.layer_n)+":"
        weights, biases = self.weights, self.biases
        assert weights.shape[1] == prev_layer.output_dims[0]
        assert weights.shape[0] == biases.shape[0]
        assert np.product(weights.shape[1:]) <= self.weight_e
        input_neurons = prev_layer.output_neurons()
        input_channels = weights.shape[1]
        output_channels = weights.shape[0]
        kernel_size = weights.shape[2:]
        side_lengths = (int((prev_layer.output_dims[1] - kernel_size[0]) / self.stride + 1), int((prev_layer.output_dims[2] - kernel_size[1]) / self.stride + 1))
        self.output_dims = (output_channels, *side_lengths, prev_layer.output_dims[-1])
        
        indices = np.arange(len(input_neurons)).reshape(prev_layer.output_dims)
        for output_channel in range(output_channels): # no of output channels is most outer loop
            patches = [image.extract_patches_2d(indices[input_channel,:,:,:], (kernel_size)) for input_channel in range(input_channels)]
            patches = np.stack(patches)
            assert np.product(side_lengths) == patches.shape[1]
            bias = quartz.blocks.ConstantDelay(value=biases[output_channel], monitor=False, name=name+"l{}-b{}-".format(self.layer_n, output_channel), parent_layer=self)
            splitter = quartz.blocks.Splitter(name=name+"l{}-bias{}-split-".format(self.layer_n, output_channel), promoted=False, monitor=False, parent_layer=self)
            bias.output_neurons()[0].connect_to(splitter.input_neurons()[0], self.weight_e, self.t_syn)
            input_neurons[0].connect_to(bias.recall_neurons()[0], self.weight_e, self.t_syn)
            self.blocks += [bias, splitter]
            for i in range(np.product(side_lengths)): # loop through all units in the output channel
                combinations = []
                for input_channel in range(0,input_channels):
                    neuron_patch = np.array(input_neurons)[patches[input_channel,i,:,:,:].flatten()]
                    combinations += list(zip(neuron_patch[::2], neuron_patch[1::2], weights[output_channel,input_channel,:,:].flatten()))
                bias_sign = 1 if biases[output_channel] >= 0 else -1
                combinations += [(splitter.first(), splitter.second(), bias_sign)]
                relco = quartz.blocks.ReLCo(combinations, split_input=True, split_output=self.split_output, monitor=monitor,\
                                  name="conv-l{0}-c{1:3.0f}-n{2:3.0f}-".format(self.layer_n, output_channel, i), parent_layer=self)
                self.blocks += [relco]
                self.neurons += relco.output_neurons()


class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride=None, name="pool:", split_output=True, monitor=False, **kwargs):
        super(MaxPool2D, self).__init__(name=name, **kwargs)
        self.name = name
        self.split_output = split_output
        self.monitor = monitor

    def connect_from(self, prev_layer):
        self.prev_layer = prev_layer
        self.layer_n = prev_layer.layer_n + 1
        self.name += str(self.layer_n)+":"
        if stride==None: stride = kernel_size[0]
        input_neurons = prev_layer.output_neurons()
        self.output_dims = list(prev_layer.output_dims)
        self.output_dims[1] = int(self.output_dims[1]/kernel_size[0])
        self.output_dims[2] = int(self.output_dims[2]/kernel_size[1])
        n_output_channels = self.output_dims[0]
        extra_delay_first = 0
        extra_delay_sec = 0
        if isinstance(prev_layer, Conv2D):
            extra_delay_sec=2
        if isinstance(prev_layer, MaxPool2D):
            extra_delay_first=1

        indices = np.arange(len(input_neurons)).reshape(prev_layer.output_dims)
        for output_channel in range(n_output_channels): # no of output channels is most outer loop
            patches = image.extract_patches_2d(indices[output_channel,:,:,:], (kernel_size)) # extract patches with stride 1
            patches = np.stack(patches)
            patches_side_length = int(np.sqrt(patches.shape[0]))
            patches = patches.reshape(patches_side_length, patches_side_length, *kernel_size, -1) # align patches as a rectangle
            patches = patches[::self.stride,::self.stride,:,:,:].reshape(-1, *kernel_size, patches.shape[-1]) # pick only patches that are interesting (stride)
            
            for i in range(int(np.product(self.output_dims[1:3]))): # loop through all units in the output channel
                neuron_patch = np.array(input_neurons)[patches[i,:,:,:].flatten()]
                combination = list(zip(neuron_patch[::2], neuron_patch[1::2],))
                
                maxpool = quartz.blocks.MaxPooling(combination, split_input=True, split_output=self.split_output, monitor=monitor,
                                                  extra_delay_first=extra_delay_first, extra_delay_sec=extra_delay_sec,
                                                  name="pool-l{0}-c{1:3.0f}-n{2:3.0f}-".format(self.layer_n, output_channel, i), parent_layer=self)
                self.blocks += [maxpool]
                self.neurons += maxpool.output_neurons()


class MonitorLayer(Layer):
    def __init__(self, name="monitor", **kwargs):
        super(MonitorLayer, self).__init__(name=name, **kwargs)

    def connect_from(self, prev_layer):
        self.layer_n = prev_layer.layer_n + 1
        self.prev_layer = prev_layer
        self.name += str(self.layer_n)+":"
        weight_e, weight_acc, t_syn, t_min, t_neu = self.get_params_at_once()
        for neuron in prev_layer.output_neurons():
            output_neuron = Neuron(name=neuron.name + "-monitor", monitor=True)
            neuron.connect_to(output_neuron, weight_e, t_syn)
            self.neurons += [output_neuron]
