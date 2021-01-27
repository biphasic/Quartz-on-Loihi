from quartz.neuron import Neuron
from quartz.blocks import Block
import quartz
from sklearn.feature_extraction import image
import numpy as np
import math
import ipdb


class Layer:
    def __init__(self, name, weight_e=30, weight_acc=255, t_min=1, t_neu=1, monitor=False):
        self.name = name
        self.weight_e = weight_e
        self.weight_acc = weight_acc
        self.t_min = t_min
        self.t_neu = t_neu
        self.monitor = monitor
        self.output_dims = []
        self.layer_n = None
        self.blocks = []
        self.output_neurons = []
        self.sync_neurons = []
        self.rectifier_neurons = []
        self.num_dendritic_accumulators = 2**3

    
    def neurons(self): return self.output_neurons + self.sync_neurons + self.rectifier_neurons
        
    def names(self): return [neuron.name for neuron in self.neurons()]
    
    def n_compartments(self):
        if isinstance(self, quartz.layers.InputLayer): return 0
        return sum([block.n_compartments() for block in self.blocks])

    def n_parameters(self):
        if isinstance(self, (quartz.layers.MaxPool2D, quartz.layers.InputLayer)): return 0
        n_params = np.product(self.weights.shape)
        if self.biases is not None: n_params += np.product(self.biases.shape)
        return n_params
    
    def n_spikes(self):
        if isinstance(self, quartz.layers.MaxPool2D):
            return int(-1 * (np.product(self.kernel_size)-1) / np.product(self.kernel_size) * np.product(self.prev_layer.output_dims) + np.product(self.output_dims))
        return self.n_compartments()

    def n_outgoing_connections(self):
        return sum([block.n_outgoing_connections() for block in self.blocks])

    def n_recurrent_connections(self):
        return sum([block.n_recurrent_connections() for block in self.blocks])

    def __repr__(self):
        return self.name


class InputLayer(Layer):
    def __init__(self, dims, name="l0-input:", **kwargs):
        super(InputLayer, self).__init__(name=name, **kwargs)
        self.layer_n = 0
        self.output_dims = dims
        self.sync_neurons += [Neuron(name=name+"sync:", type=Neuron.input)]
        self.rectifier_neurons += [Neuron(name=name+"rectifier:", type=Neuron.input)]
        for channel in range(dims[0]):
            for height in range(dims[2]):
                for width in range(dims[1]):
                    self.output_neurons += [Neuron(name=name+"input-c{}w{}h{}:".format(channel,height,width), type=Neuron.input)]


class Dense(Layer):
    def __init__(self, weights, biases=None, rectifying=True, name="dense:", **kwargs):
        super(Dense, self).__init__(name=name, **kwargs)
        self.weights = weights.copy()
        self.biases = biases
        self.rectifying = rectifying
        self.output_dims = weights.shape[0]

    def connect_from(self, prev_layer, t_max):
        self.layer_n = prev_layer.layer_n + 1
        self.name = "l{}-{}".format(self.layer_n, self.name)
        weights, biases = self.weights, self.biases
        assert weights.shape[1] == len(prev_layer.output_neurons)
        if biases is not None: assert weights.shape[0] == biases.shape[0]

        # create and connect support neurons
        sync = Neuron(name=self.name+"sync:", type=Neuron.sync)
        rectifier = Neuron(name=self.name+"rectifier:", type=Neuron.rectifier, loihi_type=Neuron.acc)
        self.sync_neurons += [sync]
        self.rectifier_neurons += [rectifier]
        prev_layer.rectifier_neurons[0].connect_to(sync, self.weight_e)
        prev_layer.rectifier_neurons[0].connect_to(rectifier, self.weight_acc)
        
        # create all neurons converted from units and create self-inhibiting group connection
        self.output_neurons = [Neuron(name=self.name+"relco-n{1:3.0f}:".format(self.layer_n, i), loihi_type=Neuron.acc) for i in range(self.weights.shape[0])]
        neuron_block = Block(name=self.name+"all-units")
        neuron_block.neurons += self.output_neurons
        neuron_block.connect_to(neuron_block, -255*np.eye(len(self.output_neurons)), 6, 0)
        self.blocks += [neuron_block]
        
        # group neurons from previous layer
        output_block = Block(name=prev_layer.name+"dense-block")
        prev_layer.blocks += [output_block]
        output_block.neurons += prev_layer.output_neurons
        
        # connect groups
        delay = 0 if isinstance(prev_layer, quartz.layers.InputLayer) else 1
        output_block.connect_to(neuron_block, self.weights*self.weight_acc, 0, delay)
        for i in range(self.output_dims):
            if biases is not None and biases[i] != 0:
                bias_sign = np.sign(biases[i])
#                 delay = round((1-biases[i])*t_max)
#                 source = prev_layer.sync_neurons[0]
#                 while delay > self.num_dendritic_accumulators
#                     source.connect_to(self.output_neurons[i], bias_sign*self.weight_acc, 0, )
#                 print(round((1-biases[i])*t_max))
            # negative sum of quantized weights to balance first spikes and bias
            bias_balance = -(bias_sign) if biases is not None and biases[i] != 0 else 0
            weight_sum = -sum((weights[i,:]*255).round()/255) + bias_balance + 1
            for _ in range(int(abs(weight_sum))):
                sync.connect_to(self.output_neurons[i], np.sign(weight_sum)*self.weight_acc)
            weight_rest = weight_sum - int(weight_sum)
            sync.connect_to(self.output_neurons[i], weight_rest*self.weight_acc)
        if self.rectifying:
            rectifier.group_connect_to(neuron_block, 251, 6, 0)


class Conv2D(Layer):
    def __init__(self, weights, biases=None, stride=(1,1), padding=(0,0), groups=1, name="conv2D:", monitor=False, **kwargs):
        super(Conv2D, self).__init__(name=name, monitor=monitor, **kwargs)
        self.weights = weights.copy()
        self.biases = biases
        if isinstance(stride, int): stride = (stride, stride)
        self.stride = stride
        if isinstance(padding, int): padding = (padding, padding)
        self.padding = padding
        self.groups = groups

    def connect_from(self, prev_layer):
        self.prev_layer = prev_layer
        self.layer_n = prev_layer.layer_n + 1
        self.name = "l{}-{}".format(self.layer_n, self.name)
        weights, biases = self.weights, self.biases
        weights = (weights*self.weight_acc).round()/self.weight_acc
        assert weights.shape[1]*self.groups == prev_layer.output_dims[0]
        if biases is not None: assert weights.shape[0] == biases.shape[0]
        input_blocks = prev_layer.output_blocks()
        output_channels, input_channels, *kernel_size = self.weights.shape
        input_channels *= self.groups
        side_lengths = (int((prev_layer.output_dims[1] - kernel_size[0] + 2*self.padding[0]) / self.stride[0] + 1), 
                        int((prev_layer.output_dims[2] - kernel_size[1] + 2*self.padding[1]) / self.stride[1] + 1))
        self.output_dims = (output_channels, *side_lengths)

        input_blocks = np.pad(np.array(input_blocks).reshape(*prev_layer.output_dims), 
                              ((0,0), self.padding, self.padding), 'constant', constant_values=(0))
        indices = np.arange(len(input_blocks.ravel())).reshape(input_blocks.shape)
        input_blocks = input_blocks.ravel()
        n_groups_out = output_channels//self.groups
        n_groups_in = input_channels//self.groups
        prev_trigger = prev_layer.trigger_blocks()[0]
        trigger_delay = 0 #if isinstance(prev_layer, quartz.layers.InputLayer) else 2
        trigger_block = quartz.blocks.Trigger(n_channels=output_channels, name=self.name+"trigger:", parent_layer=self)
        for i in range(output_channels):
            prev_trigger.rectifier_neurons[0].connect_to(trigger_block.output_neurons[i], self.weight_e, trigger_delay)
        prev_trigger.rectifier_neurons[0].connect_to(trigger_block.rectifier_neurons[0], self.weight_acc, trigger_delay)
        self.blocks += [trigger_block]
        for g in range(self.groups): # split feature maps into groups
            for output_channel in range(g*n_groups_out,(g+1)*n_groups_out): # loop over output channels in one group
                patches = [image.extract_patches_2d(indices[input_channel,:,:], (kernel_size)) for input_channel in range(input_channels)]
                patches = np.stack(patches)
                assert np.product(side_lengths) == patches.shape[1]
                if biases is not None: # create bias for output channel
                    bias = quartz.blocks.Bias(value=biases[output_channel], name=self.name+"bias-n{0:2.0f}:".format(output_channel), 
                                                       type=Block.hidden, monitor=False, parent_layer=self)
                    trigger_index = 1 if isinstance(prev_layer, (quartz.layers.InputLayer, quartz.layers.MaxPool2D)) else 0
                    prev_layer.trigger_blocks()[trigger_index].output_neurons[0].connect_to(bias.input_neurons[0], self.weight_e)
                    self.blocks += [bias]
                for i in range(np.product(side_lengths)): # loop through all units in the output channel
                    relco = quartz.blocks.ReLCo(name=self.name+"relco-c{1:3.0f}-n{2:3.0f}:".format(self.layer_n, output_channel, i), parent_layer=self)
                    self.blocks += [relco]
                    if biases is not None:
                        bias_sign = np.sign(biases[output_channel])
                        if bias_sign == 0: bias_sign = 1  # make it positive in case biases[i] == 0
                        bias.output_neurons[0].connect_to(relco.input_neurons[0], bias_sign*self.weight_acc)                   
                    weight_sum = 0
                    delay =  0 if isinstance(prev_layer, quartz.layers.InputLayer) else 1
                    for group_weight_index, input_channel in enumerate(range(g*n_groups_in,(g+1)*n_groups_in)):
                        block_patch = input_blocks[patches[input_channel,i,:,:].ravel()]
                        patch_weights = weights[output_channel,group_weight_index,:,:].ravel()
                        assert block_patch.shape[0] == patch_weights.shape[0]
                        for j, block in enumerate(block_patch):
                            if block != 0: # no connection when trying to connect to padding block
                                weight = patch_weights[j]
                                weight_sum += weight
                                block.output_neurons[0].connect_to(relco.input_neurons[0], weight*self.weight_acc, delay)
                    bias_balance = -(bias_sign - 1) if biases is not None else 1
                    weight_sum = -weight_sum + bias_balance                    
                    for _ in range(int(abs(weight_sum))):
                        trigger_block.output_neurons[output_channel].connect_to(relco.input_neurons[0], np.sign(weight_sum)*self.weight_acc, trigger_delay)
                    weight_rest = weight_sum - int(weight_sum)
                    trigger_block.output_neurons[output_channel].connect_to(relco.input_neurons[0], weight_rest*self.weight_acc, trigger_delay)
                    trigger_block.rectifier_neurons[0].connect_to(relco.neuron("calc"), 2**6*251, trigger_delay)


class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride=None, name="maxpool:", **kwargs):
        super(MaxPool2D, self).__init__(name=name, **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride

    def connect_from(self, prev_layer):
        self.prev_layer = prev_layer
        self.layer_n = prev_layer.layer_n + 1
        self.name = "l{}-{}".format(self.layer_n, self.name)
        self.output_dims = list(prev_layer.output_dims)
        output_channels = self.output_dims[0]
        if self.stride==None: self.stride = self.kernel_size[0]
        self.output_dims[1] = int(self.output_dims[1]/self.kernel_size[0])
        self.output_dims[2] = int(self.output_dims[2]/self.kernel_size[1])
        
        trigger_block = quartz.blocks.WTA(name=self.name+"trigger:", type=Block.trigger, parent_layer=self)
        trigger_block_bias = quartz.blocks.WTA(name=self.name+"rectifier:", type=Block.trigger, parent_layer=self)
        trigger_block.rectifier_neurons += [trigger_block.neurons[0]]
        trigger_block_bias.output_neurons += [trigger_block_bias.neurons[0]]
        self.blocks += [trigger_block, trigger_block_bias]
        prev_layer.trigger_blocks()[0].rectifier_neurons[0].connect_to(trigger_block.input_neurons[0], self.weight_e)
        trigger_index = 1 if isinstance(prev_layer, (quartz.layers.InputLayer, quartz.layers.MaxPool2D)) else 0
        prev_layer.trigger_blocks()[trigger_index].output_neurons[0].connect_to(trigger_block_bias.input_neurons[0], self.weight_e)
        
        input_blocks = np.array(prev_layer.output_blocks())
        indices = np.arange(len(input_blocks)).reshape(prev_layer.output_dims)
        for output_channel in range(output_channels): # no of output channels is most outer loop
            patches = image.extract_patches_2d(indices[output_channel,:,:], (self.kernel_size)) # extract patches with stride 1
            patches = np.stack(patches)
            patches_side_length = int(np.sqrt(patches.shape[0]))
            patches = patches.reshape(patches_side_length, patches_side_length, *self.kernel_size, -1) # align patches as a rectangle
            # stride patches
            patches = patches[::self.stride,::self.stride,:,:,:].reshape(-1, *self.kernel_size, patches.shape[-1]) 
            for i in range(int(np.product(self.output_dims[1:]))): # loop through all units in the output channel
                block_patch = input_blocks[patches[i,:,:,:].ravel()]
                maxpool = quartz.blocks.WTA(name=self.name+"wta-c{0:3.0f}-n{1:3.0f}:".format(output_channel, i),
                                                type=Block.output, parent_layer=self)
                maxpool.output_neurons += [maxpool.neurons[0]]
                for block in block_patch:
                    for neuron in block.output_neurons:
                        neuron.connect_to(maxpool.input_neurons[0], self.weight_e)
                        maxpool.input_neurons[0].connect_to(neuron, -64*255)
                self.blocks += [maxpool]

