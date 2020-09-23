from quartz.components import Neuron, Synapse
from quartz.blocks import Block
import quartz
from sklearn.feature_extraction import image
import numpy as np
import ipdb


class Layer:
    def __init__(self, name, weight_e=2**7, weight_acc=2**7, t_min=1, t_neu=1, monitor=False):
        self.name = name
        self.weight_e = weight_e
        self.weight_acc = weight_acc
        self.t_min = t_min
        self.t_neu = t_neu
        self.monitor = monitor
        self.output_dims = []
        self.layer_n = None
        self.prev_layer = None
        self.blocks = []
        self.compartment_groups = []

    def _get_blocks_of_type(self, block_type):
        return [block for block in self.blocks if block.type == block_type]
    
    def input_blocks(self): return self._get_blocks_of_type(Block.input)

    def output_blocks(self): return self._get_blocks_of_type(Block.output)
    
    def get_params_at_once(self):
        return self.weight_e, self.weight_acc, self.t_min, self.t_neu
    
    def neurons(self):
        return [block.neurons for block in self.blocks]

    def n_compartments(self):
        return sum([block.n_compartments() for block in self.blocks])

    def n_parameters(self):
        if isinstance(self, quartz.layers.InputLayer) or isinstance(self, quartz.layers.MonitorLayer) or isinstance(self, quartz.layers.MaxPool2D): return 0
        n_params = np.product(self.weights.shape)
        if self.biases is not None: n_params += np.product(self.biases.shape)
        return n_params
    
    def n_connections(self):
        return sum([block.n_connections() for block in self.blocks])

    def check_block_delays(self, t_max, numDendriticAccumulators):
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
        super(InputLayer, self).__init__(name=name, monitor=monitor, **kwargs)
        self.layer_n = 0
        self.output_dims = dims

        for channel in range(dims[0]):
            for height in range(dims[2]):
                for width in range(dims[1]):
                    splitter = quartz.blocks.Splitter(name=name+"split-c{}w{}h{}".format(channel,height,width), 
                                                      type=Block.output, monitor=monitor, parent_layer=self)
                    self.blocks.append(splitter)


class Dense(Layer):
    def __init__(self, weights, biases, name="dense:", monitor=False, **kwargs):
        super(Dense, self).__init__(name=name, monitor=monitor, **kwargs)
        self.weights = weights
        self.biases = biases
        self.output_dims = weights.shape[0]

    def connect_from(self, prev_layer):
        self.prev_layer = prev_layer
        self.layer_n = prev_layer.layer_n + 1
        self.name = "l{}-{}".format(self.layer_n, self.name)
        weights, biases = self.weights, self.biases
        if biases is not None: assert weights.shape[0] == biases.shape[0]

        input_blocks = prev_layer.output_blocks()
        assert weights.shape[1] == len(input_blocks)
        assert len(input_blocks) <= self.weight_e
        n_inputs = len(input_blocks) if biases is None else len(input_blocks) + 1
        for i in range(self.output_dims):
            relco = quartz.blocks.ReLCo(name=self.name+"l{0:1.0f}-n{1:3.0f}".format(self.layer_n, i), monitor=self.monitor, parent_layer=self)
            if biases is not None:
                bias = quartz.blocks.ConstantDelay(value=biases[i], name=self.name+"l{0}-b{1}".format(self.layer_n, i), type=Block.hidden, monitor=False, parent_layer=self)
                splitter = quartz.blocks.Splitter(name=self.name+"l{0}-bias{1}-split".format(self.layer_n, i), type=Block.hidden, monitor=False, parent_layer=self)
                bias.connect_to(splitter, self.weight_e)
                input_blocks[0].connect_to(bias, np.array([[self.weight_e, 0]]))
                bias_sign = 1 if biases[i] >= 0 else -1
                splitter.connect_to(relco, np.array([[bias_sign,self.weight_e/n_inputs],[-bias_sign, 0]]))
                self.blocks += [bias, splitter]
            
            for j, block in enumerate(input_blocks):
                weight = weights[i,j]
                block.connect_to(relco, weight=np.array([[weight,self.weight_e/n_inputs],[-weight, 0]]))
            self.blocks += [relco]
        #ipdb.set_trace()


class Conv2D(Layer):
    def __init__(self, weights, biases, stride=1, name="conv2D:", monitor=False, **kwargs):
        super(Conv2D, self).__init__(name=name, monitor=monitor, **kwargs)
        self.weights = weights
        self.biases = biases
        self.stride = stride

    def connect_from(self, prev_layer):
        self.prev_layer = prev_layer
        self.layer_n = prev_layer.layer_n + 1
        self.name = "l{}-{}".format(self.layer_n, self.name)
        weights, biases = self.weights, self.biases
        assert weights.shape[1] == prev_layer.output_dims[0]
        assert weights.shape[0] == biases.shape[0]
        assert np.product(weights.shape[1:]) <= self.weight_e
        input_neurons = prev_layer.output_neurons()
        input_channels = weights.shape[1]
        output_channels = weights.shape[0]
        kernel_size = weights.shape[2:]
        side_lengths = (int((prev_layer.output_dims[1] - kernel_size[0]) / self.stride + 1), int((prev_layer.output_dims[2] - kernel_size[1]) / self.stride + 1))
        self.output_dims = (output_channels, *side_lengths, 2) if self.split_output else (output_channels, *side_lengths, 1)
        
        indices = np.arange(len(input_neurons)).reshape(prev_layer.output_dims)
        for output_channel in range(output_channels): # no of output channels is most outer loop
            patches = [image.extract_patches_2d(indices[input_channel,:,:,:], (kernel_size)) for input_channel in range(input_channels)]
            patches = np.stack(patches)
            assert np.product(side_lengths) == patches.shape[1]
            bias = quartz.blocks.ConstantDelay(value=biases[output_channel], monitor=False, name=self.name+"l{}-b{}-".format(self.layer_n, output_channel), parent_layer=self)
            splitter = quartz.blocks.Splitter(name=self.name+"l{}-bias{}-split-".format(self.layer_n, output_channel), promoted=False, monitor=False, parent_layer=self)
            bias.output_neurons()[0].connect_to(splitter.input_neurons()[0], self.weight_e)
            input_neurons[0].connect_to(bias.recall_neurons()[0], self.weight_e)
            self.blocks += [bias, splitter]
            for i in range(np.product(side_lengths)): # loop through all units in the output channel
                combinations = []
                for input_channel in range(0,input_channels):
                    neuron_patch = np.array(input_neurons)[patches[input_channel,i,:,:,:].flatten()]
                    combinations += list(zip(neuron_patch[::2], neuron_patch[1::2], weights[output_channel,input_channel,:,:].flatten()))
                bias_sign = 1 if biases[output_channel] >= 0 else -1
                combinations += [(splitter.first(), splitter.second(), bias_sign)]
                relco = quartz.blocks.ReLCo(combinations, split_input=True, split_output=self.split_output, monitor=self.monitor,\
                                  name=self.name+"l{0}-c{1:3.0f}-n{2:3.0f}".format(self.layer_n, output_channel, i), parent_layer=self)
                self.blocks += [relco]
                self.neurons += relco.output_neurons()


class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride=None, name="pool:", monitor=False, **kwargs):
        super(MaxPool2D, self).__init__(name=name, monitor=monitor, **kwargs)
        self.kernel_size = kernel_size
        self.split_output = split_output
        self.stride = stride

    def connect_from(self, prev_layer):
        self.prev_layer = prev_layer
        self.layer_n = prev_layer.layer_n + 1
        self.name = "l{}-{}".format(self.layer_n, self.name)
        kernel_size = self.kernel_size
        if self.stride==None: self.stride = kernel_size[0]
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
                
                maxpool = quartz.blocks.MaxPooling(combination, split_input=True, split_output=self.split_output, monitor=self.monitor,
                                                  extra_delay_first=extra_delay_first, extra_delay_sec=extra_delay_sec,
                                                  name="pool-l{0}-c{1:3.0f}-n{2:3.0f}".format(self.layer_n, output_channel, i), parent_layer=self)
                self.blocks += [maxpool]
                self.neurons += maxpool.output_neurons()


class MonitorLayer(Layer):
    def __init__(self, name="monitor:", monitor=True, **kwargs):
        super(MonitorLayer, self).__init__(name=name, monitor=monitor, **kwargs)

    def connect_from(self, prev_layer):
        self.layer_n = prev_layer.layer_n + 1
        self.prev_layer = prev_layer
        self.name = "l{}-{}".format(self.layer_n, self.name)
        
        for block in prev_layer.output_blocks():
            monitor = quartz.blocks.Block(name=self.name, type=Block.output, monitor=self.monitor, parent_layer=self)
            output_neuron = Neuron(name=block.name + "-monitor", type=Block.input, monitor=self.monitor)
            monitor.neurons += [output_neuron]
            self.blocks += [monitor]
            block.connect_to(monitor, np.array([[self.weight_e, self.weight_e]])) # missing delays
