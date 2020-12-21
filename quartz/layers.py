from quartz.components import Neuron, Synapse
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
        self.prev_layer = None
        self.blocks = []

    def _get_blocks_of_type(self, block_type):
        return [block for block in self.blocks if block.type == block_type]
    
    def input_blocks(self): return self._get_blocks_of_type(Block.input)

    def output_blocks(self): return self._get_blocks_of_type(Block.output)

    def trigger_blocks(self): return self._get_blocks_of_type(Block.trigger)
    
    def get_params_at_once(self):
        return self.weight_e, self.weight_acc, self.t_min, self.t_neu
    
    def neurons(self): return [block.neurons for block in self.blocks]
    
    def names(self): return [block.name for block in self.blocks]
    
    def n_compartments(self):
        if isinstance(self, quartz.layers.InputLayer) or isinstance(self, quartz.layers.MonitorLayer): return 0
        return sum([block.n_compartments() for block in self.blocks])

    def n_parameters(self):
        if isinstance(self, quartz.layers.InputLayer) or isinstance(self, quartz.layers.MonitorLayer): return 0
        n_params = np.product(self.weights.shape)
        if self.biases is not None: n_params += np.product(self.biases.shape)
        return n_params
    
    def n_spikes(self):
        saved_spikes = 0
        for block in self.blocks:
            if isinstance(block, quartz.blocks.ConvMax):
                saved_spikes += len(block.neurons) - 2
        return self.n_compartments() - saved_spikes

    def n_outgoing_connections(self):
        return sum([block.n_outgoing_connections() for block in self.blocks])

    def n_recurrent_connections(self):
        return sum([block.n_recurrent_connections() for block in self.blocks])

    def check_block_delays(self, t_max, numDendriticAccumulators):
        for block in self.blocks:
            if isinstance(block, quartz.blocks.ConstantDelay):
                if not block.layout: block.layout_delays(t_max, numDendriticAccumulators)

    def print_connections(self, maximum=10e7):
        for i, block in enumerate(self.blocks):
            block.print_connections()
            if i > maximum: break

    def __repr__(self):
        return self.name

    
class InputLayer(Layer):
    def __init__(self, dims, name="l0-input:", **kwargs):
        super(InputLayer, self).__init__(name=name, **kwargs)
        self.layer_n = 0
        self.output_dims = dims
        trigger_block = quartz.blocks.Input(name=name+"trigger:", type=Block.trigger, parent_layer=self)
        self.blocks += [trigger_block]
        for channel in range(dims[0]):
            for height in range(dims[2]):
                for width in range(dims[1]):
                    pixel = quartz.blocks.Input(name=name+"input-c{}w{}h{}:".format(channel,height,width), parent_layer=self)
                    self.blocks += [pixel]
        

class Dense(Layer):
    def __init__(self, weights, biases, rectifying=True, name="dense:", **kwargs):
        super(Dense, self).__init__(name=name, **kwargs)
        self.weights = weights.copy()
        self.biases = biases
        self.rectifying = rectifying
        self.output_dims = weights.shape[0]

    def connect_from(self, prev_layer):
        self.prev_layer = prev_layer
        self.layer_n = prev_layer.layer_n + 1
        self.name = "l{}-{}".format(self.layer_n, self.name)
        weights, biases = self.weights, self.biases
        input_blocks = prev_layer.output_blocks()
        assert weights.shape[1] == len(input_blocks)
        if biases is not None: assert weights.shape[0] == biases.shape[0]
        prev_trigger = prev_layer.trigger_blocks()[0]
        trigger_block = quartz.blocks.Trigger(n_channels=1, name=self.name+"trigger:", parent_layer=self)
        trigger_delay = 0 if isinstance(prev_layer, quartz.layers.InputLayer) else 2
        prev_trigger.output_neurons[0].connect_to(trigger_block.output_neurons[0], self.weight_acc, trigger_delay)
        self.blocks += [trigger_block]
        for i in range(self.output_dims):
            if self.rectifying:
                relco = quartz.blocks.ReLCo(name=self.name+"relco-n{1:3.0f}:".format(self.layer_n, i), 
                                            monitor=self.monitor, parent_layer=self)
            else:
                relco = quartz.blocks.Output(name=self.name+"output-n{1:3.0f}:".format(self.layer_n, i), 
                                            monitor=self.monitor, parent_layer=self)
            for j, block in enumerate(input_blocks):
                weight = weights[i,j]
                delay = 0 #delay = 5 if weight > 0 else 0
                block.first().connect_to(relco.neuron("calc"), weight*self.weight_acc, delay+self.t_min)
            self.blocks += [relco]
            # negative sum of quantized weights to balance first spikes and  +1 is for readout
            weight_sum = -sum((weights[i,:]*255).round()/255) + 1
            for _ in range(int(abs(weight_sum))):
                trigger_block.output_neurons[0].connect_to(relco.neuron("calc"), np.sign(weight_sum)*self.weight_acc, delay)
            weight_rest = weight_sum - int(weight_sum)
            trigger_block.output_neurons[0].connect_to(relco.neuron("calc"), weight_rest*self.weight_acc, delay)
            trigger_block.rectifier_neurons[0].connect_to(relco.neuron("1st"), self.weight_e, delay)
            if biases is not None:
                bias = quartz.blocks.ConstantDelay(value=biases[i], name=self.name+"const-n{0:2.0f}:".format(i), 
                                                   type=Block.hidden, parent_layer=self)
                splitter = quartz.blocks.Splitter(name=self.name+"split-bias-n{0:2.0f}:".format(i), 
                                                  type=Block.hidden, parent_layer=self)
                bias.output_neurons[0].connect_to(splitter.input_neurons[0], self.weight_e)
                prev_trigger.output_neurons[0].connect_to(bias.input_neurons[0], self.weight_e, trigger_delay) # possibly be smarter about this one
                bias_sign = np.sign(biases[i])
                splitter.first().connect_to(relco.input_neurons[0], bias_sign*self.weight_acc, self.t_min)
                splitter.second().connect_to(relco.input_neurons[0], -bias_sign*self.weight_acc)
                self.blocks += [bias, splitter]


class Conv2D(Layer):
    def __init__(self, weights, biases, stride=(1,1), padding=(0,0), groups=1, name="conv2D:", monitor=False, **kwargs):
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
        indices = np.arange(len(input_blocks.flatten())).reshape(input_blocks.shape)
        n_groups_out = output_channels//self.groups
        n_groups_in = input_channels//self.groups
        prev_trigger = prev_layer.trigger_blocks()[0]
        trigger_delay = 0 if isinstance(prev_layer, quartz.layers.InputLayer) else 2
        trigger_block = quartz.blocks.Trigger(n_channels=output_channels, name=self.name+"trigger:", parent_layer=self)
        for i in range(output_channels):
            prev_trigger.output_neurons[0].connect_to(trigger_block.output_neurons[i], self.weight_acc, trigger_delay)
        self.blocks += [trigger_block]
        for g in range(self.groups): # split feature maps into groups
            for output_channel in range(g*n_groups_out,(g+1)*n_groups_out): # loop over output channels in one group
                patches = [image.extract_patches_2d(indices[input_channel,:,:], (kernel_size)) for input_channel in range(input_channels)]
                patches = np.stack(patches)
                assert np.product(side_lengths) == patches.shape[1]
                if biases is not None:
                    bias = quartz.blocks.ConstantDelay(value=biases[output_channel], name=self.name+"const-n{0:2.0f}:".format(output_channel), 
                                                       type=Block.hidden, monitor=False, parent_layer=self)
                    splitter = quartz.blocks.Splitter(name=self.name+"split-bias-n{0:2.0f}:".format(output_channel), 
                                                      type=Block.hidden, monitor=False, parent_layer=self)
                    bias.output_neurons[0].connect_to(splitter.input_neurons[0], self.weight_e)
                    prev_trigger.output_neurons[0].connect_to(bias.input_neurons[0], self.weight_e, trigger_delay) 
                    self.blocks += [bias, splitter]
                for i in range(np.product(side_lengths)): # loop through all units in the output channel
                    relco = quartz.blocks.ReLCo(name=self.name+"relco-c{1:3.0f}-n{2:3.0f}:".format(self.layer_n, output_channel, i), parent_layer=self)
                    self.blocks += [relco]
                    weight_sum = 0
                    delay = 0 # 4 if weight > 0 else 0
                    for group_weight_index, input_channel in enumerate(range(g*n_groups_in,(g+1)*n_groups_in)):
                        block_patch = input_blocks.flatten()[patches[input_channel,i,:,:].flatten()]
                        patch_weights = weights[output_channel,group_weight_index,:,:].flatten()
                        assert len(block_patch) == len(patch_weights)
                        for j, block in enumerate(block_patch):
                            if block != 0: # no connection when trying to connect to padding block
                                weight = patch_weights[j]
                                weight_sum += weight
                                block.first().connect_to(relco.input_neurons[0], weight*self.weight_acc, delay+self.t_min)
                    weight_sum = -weight_sum + 1
                    for _ in range(int(abs(weight_sum))):
                        trigger_block.output_neurons[output_channel].connect_to(relco.neuron("calc"), np.sign(weight_sum)*self.weight_acc, delay)
                    weight_rest = weight_sum - int(weight_sum)
                    trigger_block.output_neurons[output_channel].connect_to(relco.neuron("calc"), weight_rest*self.weight_acc, delay)
                    trigger_block.rectifier_neurons[0].connect_to(relco.neuron("1st"), self.weight_e, delay)
                    if biases is not None:
                        bias_sign = np.sign(biases[output_channel])
                        splitter.first().connect_to(relco.input_neurons[0], bias_sign*self.weight_acc, self.t_min)
                        splitter.second().connect_to(relco.input_neurons[0], -bias_sign*self.weight_acc)


class ConvPool2D(Layer):
    def __init__(self, weights, biases, pool_kernel_size, pool_stride=None, conv_stride=1, name="convpool:", **kwargs):
        super(ConvPool2D, self).__init__(name=name, **kwargs)
        self.weights = weights
        self.biases = biases
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.conv_stride = conv_stride

    def connect_from(self, prev_layer):
        self.prev_layer = prev_layer
        self.layer_n = prev_layer.layer_n + 1
        self.name = "l{}-{}".format(self.layer_n, self.name)
        weights, biases = self.weights, self.biases
        weights = (weights*self.weight_acc).round()/self.weight_acc
        assert self.weights.shape[1] == prev_layer.output_dims[0]
        if self.biases is not None: assert self.weights.shape[0] == self.biases.shape[0]
        n_inputs = np.product(self.weights.shape[1:])
        output_channels, input_channels, *conv_kernel_size = self.weights.shape
        side_lengths = (int((prev_layer.output_dims[1] - conv_kernel_size[0]) / self.conv_stride + 1),\
                        int((prev_layer.output_dims[2] - conv_kernel_size[1]) / self.conv_stride + 1))
        prev_trigger = prev_layer.trigger_blocks()[0]
        trigger_delay = 0 if isinstance(prev_layer, quartz.layers.InputLayer) else 2
        trigger_block = quartz.blocks.Trigger(n_channels=output_channels, name=self.name+"trigger:", parent_layer=self)
        for i in range(output_channels):
            prev_trigger.output_neurons[0].connect_to(trigger_block.output_neurons[i], self.weight_acc, trigger_delay)
        self.blocks += [trigger_block]
        
        conv_neurons = []
        input_blocks = prev_layer.output_blocks()
        indices = np.arange(len(input_blocks)).reshape(*prev_layer.output_dims)
        input_blocks = np.array(input_blocks)
        for output_channel in range(output_channels): # no of output channels is most outer loop
            patches = [image.extract_patches_2d(indices[input_channel,:,:], (conv_kernel_size)) for input_channel in range(input_channels)]
            patches = np.stack(patches)
            assert np.product(side_lengths) == patches.shape[1]
            if self.biases is not None:
                bias = quartz.blocks.ConstantDelay(value=self.biases[output_channel], name=self.name+"const-n{0:2.0f}:".format(output_channel), 
                                                   type=Block.hidden, parent_layer=self)
                splitter = quartz.blocks.Splitter(name=self.name+"split-bias-n{0:2.0f}:".format(output_channel), 
                                                  type=Block.hidden, parent_layer=self)
                bias.output_neurons[0].connect_to(splitter.input_neurons[0], self.weight_e)
                prev_trigger.output_neurons[0].connect_to(bias.input_neurons[0], self.weight_e, trigger_delay) # trigger biases not just from one block
                self.blocks += [bias, splitter]
            for i in range(np.product(side_lengths)): # loop through all units in the output channel
                calc_neuron = Neuron(name=self.name + "calc-n{0:3.0f}".format(i), loihi_type=Neuron.acc)
                conv_neurons += [calc_neuron]
                for input_channel in range(input_channels):
                    block_patch = input_blocks[patches[input_channel,i,:,:].flatten()]
                    patch_weights = self.weights[output_channel,input_channel,:,:].flatten()
                    assert len(block_patch) == len(patch_weights)
                    for j, block in enumerate(block_patch):
                        weight = patch_weights[j]
                        delay = 0 # 4 if weight > 0 else 0
                        block.first().connect_to(calc_neuron, weight*self.weight_acc, delay+self.t_min)
                weight_sum = -np.sum(weights[output_channel,:,:,:]) + 1
                for _ in range(int(abs(weight_sum))):
                    trigger_block.output_neurons[output_channel].connect_to(calc_neuron, np.sign(weight_sum)*self.weight_acc, delay)
                weight_rest = weight_sum - int(weight_sum)
                trigger_block.output_neurons[output_channel].connect_to(calc_neuron, weight_rest*self.weight_acc, delay)
                if self.biases is not None:
                    bias_sign = np.sign(self.biases[output_channel])
                    splitter.first().connect_to(calc_neuron, bias_sign*self.weight_acc, self.t_min)
                    splitter.second().connect_to(calc_neuron, -bias_sign*self.weight_acc)

        self.output_dims = [output_channels, *side_lengths]
        if self.pool_stride==None: self.pool_stride = self.pool_kernel_size[0]
        self.output_dims[1] = int(self.output_dims[1]/self.pool_kernel_size[0])
        self.output_dims[2] = int(self.output_dims[2]/self.pool_kernel_size[1])
        indices = np.arange(len(conv_neurons)).reshape(output_channels, *side_lengths)
        conv_neurons = np.array(conv_neurons)
        for output_channel in range(output_channels): # no of output channels is most outer loop
            patches = image.extract_patches_2d(indices[output_channel,:,:], (self.pool_kernel_size)) # extract patches with stride 1
            patches = np.stack(patches)
            patches_side_length = int(np.sqrt(patches.shape[0]))
            patches = patches.reshape(patches_side_length, patches_side_length, *self.pool_kernel_size, -1) # align patches as a rectangle
            # pick only patches that are interesting (stride)
            patches = patches[::self.pool_stride,::self.pool_stride,:,:,:].reshape(-1, *self.pool_kernel_size, patches.shape[-1]) 
            for i in range(int(np.product(self.output_dims[1:3]))): # loop through all units in the output channel
                neuron_patch = conv_neurons[patches[i,:,:,:].flatten()]
                maxpool = quartz.blocks.ConvMax(list(neuron_patch), name=self.name+"convmax-c{0:3.0f}-n{1:3.0f}:".format(output_channel, i),
                                                type=Block.output, parent_layer=self)
                trigger_block.rectifier_neurons[0].connect_to(maxpool.neuron("1st"), self.weight_e, delay)
                self.blocks += [maxpool]


class MonitorLayer(Layer):
    def __init__(self, name="monitor:", **kwargs):
        super(MonitorLayer, self).__init__(name=name, **kwargs)

    def connect_from(self, prev_layer):
        self.layer_n = prev_layer.layer_n + 1
        self.prev_layer = prev_layer
        self.name = "l{}-{}".format(self.layer_n, self.name)
        self.output_dims = prev_layer.output_dims
        prev_trigger = prev_layer.trigger_blocks()[0]
        trigger_block = quartz.blocks.Trigger(n_channels=1, name=self.name+"trigger:", parent_layer=self)
        trigger_delay = 0 if isinstance(prev_layer, quartz.layers.InputLayer) else 2
        prev_trigger.output_neurons[0].connect_to(trigger_block.output_neurons[0], self.weight_acc, trigger_delay)
        self.blocks += [trigger_block]

        for i, block in enumerate(prev_layer.output_blocks()):
            monitor = quartz.blocks.Block(name=block.name+"monitor", type=Block.output, monitor=self.monitor, parent_layer=self)
            output_neuron = Neuron(name=block.name + "monitor-{0:3.0f}".format(i), type=Block.output, monitor=self.monitor, parent=monitor)
            monitor.neurons += [output_neuron]
            self.blocks += [monitor]
            block.first().connect_to(output_neuron, self.weight_e)
            trigger_block.output_neurons[0].connect_to(output_neuron, self.weight_e)
