from quartz.components import Neuron, Synapse
from quartz.blocks import Block
import quartz
from sklearn.feature_extraction import image
import numpy as np
import math
import ipdb


class Layer:
    def __init__(self, name, weight_e=80, weight_acc=255, t_min=1, t_neu=1, monitor=False):
        self.name = name
        self.weight_e = weight_e
        self.weight_acc = weight_acc
        self.t_min = t_min
        self.t_neu = t_neu
        self.monitor = monitor
        self.output_dims = []
        self.layer_n = None
        self.prev_layer = None
        self.weight_scaling = 1
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
        trigger_block = quartz.blocks.Trigger(number=1, name=self.name+"trigger:", parent_layer=self)
        for channel in range(dims[0]):
            for height in range(dims[2]):
                for width in range(dims[1]):
                    splitter = quartz.blocks.Splitter(name=name+"split-c{}w{}h{}:".format(channel,height,width), 
                                                      type=Block.output, parent_layer=self)
                    self.blocks += [splitter]
        self.blocks += [trigger_block]


class Dense(Layer):
    def __init__(self, weights, biases, rectifying=True, name="dense:", **kwargs):
        super(Dense, self).__init__(name=name, **kwargs)
        self.weights = weights.copy()
        self.biases = biases
        self.rectifying = rectifying
        self.output_dims = weights.shape[0]
#         self.weight_scaling = 2**math.floor(np.log2(1/abs(weights).max())) # can only scale by power of 2
#         if self.weight_scaling > 2: self.weight_scaling = 2
#         self.weights *= self.weight_scaling

    def connect_from(self, prev_layer):
        self.prev_layer = prev_layer
        self.layer_n = prev_layer.layer_n + 1
        self.name = "l{}-{}".format(self.layer_n, self.name)
        weights, biases = self.weights, self.biases
        input_blocks = prev_layer.output_blocks()
        assert weights.shape[1] == len(input_blocks)
        n_inputs = len(input_blocks)
        self.weight_e = len(input_blocks)
        if biases is not None: 
            self.weight_e += 1
            n_inputs += 1
            assert weights.shape[0] == biases.shape[0]
        if self.weight_e < 30: self.weight_e *= 8
        for i in range(self.output_dims):
            if self.rectifying:
                relco = quartz.blocks.ReLCo(name=self.name+"relco-n{1:3.0f}:".format(self.layer_n, i), 
                                            monitor=self.monitor, parent_layer=self)
            else:
                relco = quartz.blocks.Output(name=self.name+"output-n{1:3.0f}:".format(self.layer_n, i), 
                                            monitor=self.monitor, parent_layer=self)
            for j, block in enumerate(input_blocks):
                weight = weights[i,j]
                delay = 5 if weight > 0 else 0
                block.first().connect_to(relco.input_neurons()[0], weight*self.weight_acc, delay + self.t_min)
                block.second().connect_to(relco.input_neurons()[0], -weight*self.weight_acc, delay)
                block.second().connect_to(relco.input_neurons()[1], self.weight_e/n_inputs, delay)
            self.blocks += [relco]
            if biases is not None:
                bias = quartz.blocks.ConstantDelay(value=biases[i], name=self.name+"const-n{0:2.0f}:".format(i), 
                                                   type=Block.hidden, parent_layer=self)
                splitter = quartz.blocks.Splitter(name=self.name+"split-bias-n{0:2.0f}:".format(i), 
                                                  type=Block.hidden, parent_layer=self)
                bias.output_neurons()[0].connect_to(splitter.input_neurons()[0], self.weight_e)
                input_blocks[0].output_neurons()[0].connect_to(bias.input_neurons()[0], self.weight_e) # possibly be smarter about this one
                bias_sign = np.sign(biases[i])
                splitter.first().connect_to(relco.input_neurons()[0], bias_sign*self.weight_acc, self.t_min)
                splitter.second().connect_to(relco.input_neurons()[0], -bias_sign*self.weight_acc)
                splitter.second().connect_to(relco.input_neurons()[1], self.weight_e/n_inputs)
                self.blocks += [bias, splitter]


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
        assert self.weights.shape[1] == prev_layer.output_dims[0]
        if self.biases is not None: assert self.weights.shape[0] == self.biases.shape[0]
        n_inputs = np.product(self.weights.shape[1:])
        output_channels, input_channels, *conv_kernel_size = self.weights.shape
        side_lengths = (int((prev_layer.output_dims[1] - conv_kernel_size[0]) / self.conv_stride + 1),\
                        int((prev_layer.output_dims[2] - conv_kernel_size[1]) / self.conv_stride + 1))
        prev_trigger = prev_layer.trigger_blocks()[0]
        trigger_block = quartz.blocks.Trigger(number=output_channels, name=self.name+"trigger:", parent_layer=self)
        prev_trigger.output_neurons()[0].connect_to(trigger_block.neurons[0], self.weight_e)
        self.blocks += [trigger_block]
        
        conv_neurons = []
        input_blocks = prev_layer.output_blocks()
        indices = np.arange(len(input_blocks)).reshape(*prev_layer.output_dims)
        for output_channel in range(output_channels): # no of output channels is most outer loop
            patches = [image.extract_patches_2d(indices[input_channel,:,:], (conv_kernel_size)) for input_channel in range(input_channels)]
            patches = np.stack(patches)
            assert np.product(side_lengths) == patches.shape[1]
            if self.biases is not None:
                bias = quartz.blocks.ConstantDelay(value=self.biases[output_channel], name=self.name+"const-n{0:2.0f}:".format(output_channel), 
                                                   type=Block.hidden, parent_layer=self)
                splitter = quartz.blocks.Splitter(name=self.name+"split-bias-n{0:2.0f}:".format(output_channel), 
                                                  type=Block.hidden, parent_layer=self)
                bias.output_neurons()[0].connect_to(splitter.input_neurons()[0], self.weight_e)
                input_blocks[0].output_neurons()[0].connect_to(bias.input_neurons()[0], self.weight_e) # trigger biases not just from one block
                self.blocks += [bias, splitter]
            for i in range(np.product(side_lengths)): # loop through all units in the output channel
                calc_neuron = Neuron(name=self.name + "calc-n{0:3.0f}".format(i), loihi_type=Neuron.acc)
                conv_neurons += [calc_neuron]
                trigger_block.output_neurons()[output_channel].connect_to(calc_neuron, self.weight_acc)
                for input_channel in range(input_channels):
                    block_patch = np.array(input_blocks)[patches[input_channel,i,:,:].flatten()]
                    patch_weights = self.weights[output_channel,input_channel,:,:].flatten()
                    assert len(block_patch) == len(patch_weights)
                    for j, block in enumerate(block_patch):
                        weight = patch_weights[j]
                        delay = 4 if weight > 0 else 0
                        extra_delay_second = 2 if isinstance(block, quartz.blocks.ConvMax) else 0
                        block.first().connect_to(calc_neuron, weight*self.weight_acc, delay+self.t_min)
                        block.second().connect_to(calc_neuron, -weight*self.weight_acc, delay+extra_delay_second)
                if self.biases is not None:
                    bias_sign = np.sign(self.biases[output_channel])
                    splitter.first().connect_to(calc_neuron, bias_sign*self.weight_acc, self.t_min)
                    splitter.second().connect_to(calc_neuron, -bias_sign*self.weight_acc)

        self.output_dims = [output_channels, *side_lengths]
        if self.pool_stride==None: self.pool_stride = self.pool_kernel_size[0]
        self.output_dims[1] = int(self.output_dims[1]/self.pool_kernel_size[0])
        self.output_dims[2] = int(self.output_dims[2]/self.pool_kernel_size[1])
        indices = np.arange(len(conv_neurons)).reshape(output_channels, *side_lengths)
        for output_channel in range(output_channels): # no of output channels is most outer loop
            patches = image.extract_patches_2d(indices[output_channel,:,:], (self.pool_kernel_size)) # extract patches with stride 1
            patches = np.stack(patches)
            patches_side_length = int(np.sqrt(patches.shape[0]))
            patches = patches.reshape(patches_side_length, patches_side_length, *self.pool_kernel_size, -1) # align patches as a rectangle
            # pick only patches that are interesting (stride)
            patches = patches[::self.pool_stride,::self.pool_stride,:,:,:].reshape(-1, *self.pool_kernel_size, patches.shape[-1]) 
            for i in range(int(np.product(self.output_dims[1:3]))): # loop through all units in the output channel
                neuron_patch = np.array(conv_neurons)[patches[i,:,:,:].flatten()]
                maxpool = quartz.blocks.ConvMax(list(neuron_patch), name=self.name+"convmax-c{0:3.0f}-n{1:3.0f}:".format(output_channel, i),
                                                type=Block.output, parent_layer=self)
                trigger_block.output_neurons()[output_channel].connect_to(maxpool.second(), self.weight_acc)
                self.blocks += [maxpool]


class Conv2D(Layer):
    def __init__(self, weights, biases, stride=1, name="conv2D:", monitor=False, **kwargs):
        super(Conv2D, self).__init__(name=name, monitor=monitor, **kwargs)
        self.weights = weights.copy()
        self.biases = biases
        self.stride = stride
#         self.weight_scaling = 2**math.floor(np.log2(1/abs(weights).max())) # can only scale by power of 2
#         if self.weight_scaling > 2: self.weight_scaling = 2
#         self.weights *= self.weight_scaling

    def connect_from(self, prev_layer):
        self.prev_layer = prev_layer
        self.layer_n = prev_layer.layer_n + 1
        self.name = "l{}-{}".format(self.layer_n, self.name)
        weights, biases = self.weights, self.biases
        assert weights.shape[1] == prev_layer.output_dims[0]
        n_inputs = np.product(self.weights.shape[1:])
        self.weight_e = np.product(self.weights.shape[1:])
        if biases is not None: 
            assert weights.shape[0] == biases.shape[0]
            self.weight_e += 1
            n_inputs += 1
        if self.weight_e < 40: self.weight_e *= 4
        assert self.weight_e <= 255
        input_blocks = prev_layer.output_blocks()
        output_channels, input_channels, *kernel_size = self.weights.shape
        side_lengths = (int((prev_layer.output_dims[1] - kernel_size[0]) / self.stride + 1), 
                        int((prev_layer.output_dims[2] - kernel_size[1]) / self.stride + 1))
        self.output_dims = (output_channels, *side_lengths)
        
        indices = np.arange(len(input_blocks)).reshape(*prev_layer.output_dims)
        for output_channel in range(output_channels): # no of output channels is most outer loop
            patches = [image.extract_patches_2d(indices[input_channel,:,:], (kernel_size)) for input_channel in range(input_channels)]
            patches = np.stack(patches)
            assert np.product(side_lengths) == patches.shape[1]
            if biases is not None:
                bias = quartz.blocks.ConstantDelay(value=biases[output_channel], name=self.name+"const-n{0:2.0f}:".format(output_channel), 
                                                   type=Block.hidden, monitor=False, parent_layer=self)
                splitter = quartz.blocks.Splitter(name=self.name+"split-bias-n{0:2.0f}:".format(output_channel), 
                                                  type=Block.hidden, monitor=False, parent_layer=self)
                bias.output_neurons()[0].connect_to(splitter.input_neurons()[0], self.weight_e)
                input_blocks[0].output_neurons()[0].connect_to(bias.input_neurons()[0], self.weight_e) 
                self.blocks += [bias, splitter]
            for i in range(np.product(side_lengths)): # loop through all units in the output channel
                relco = quartz.blocks.ReLCo(name=self.name+"relco-c{1:3.0f}-n{2:3.0f}:".format(self.layer_n, output_channel, i), parent_layer=self)
                self.blocks += [relco]
                for input_channel in range(input_channels):
                    block_patch = np.array(input_blocks)[patches[input_channel,i,:,:].flatten()]
                    patch_weights = weights[output_channel,input_channel,:,:].flatten()
                    assert len(block_patch) == len(patch_weights)
                    for j, block in enumerate(block_patch):
                        weight = patch_weights[j]
                        delay = 4 if weight > 0 else 0
                        extra_delay_second = 2 if isinstance(block, quartz.blocks.ConvMax) else 0
                        block.first().connect_to(relco.input_neurons()[0], weight*self.weight_acc, delay+self.t_min)
                        block.second().connect_to(relco.input_neurons()[0], -weight*self.weight_acc, delay+extra_delay_second)
                        block.second().connect_to(relco.input_neurons()[1], self.weight_e/n_inputs, delay+extra_delay_second)
                if biases is not None:
                    bias_sign = np.sign(biases[output_channel])
                    splitter.first().connect_to(relco.input_neurons()[0], bias_sign*self.weight_acc, self.t_min)
                    splitter.second().connect_to(relco.input_neurons()[0], -bias_sign*self.weight_acc)
                    splitter.second().connect_to(relco.input_neurons()[1], self.weight_e/n_inputs)


class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride=None, name="pool:", monitor=False, **kwargs):
        super(MaxPool2D, self).__init__(name=name, monitor=monitor, **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride

    def connect_from(self, prev_layer):
        self.prev_layer = prev_layer
        self.layer_n = prev_layer.layer_n + 1
        self.name = "l{}-{}".format(self.layer_n, self.name)
        kernel_size = self.kernel_size
        if self.stride==None: self.stride = kernel_size[0]
        input_blocks = prev_layer.output_blocks()
        self.weight_e = np.product(kernel_size) * 10
        self.output_dims = list(prev_layer.output_dims)
        self.output_dims[1] = int(self.output_dims[1]/kernel_size[0])
        self.output_dims[2] = int(self.output_dims[2]/kernel_size[1])
        n_output_channels = self.output_dims[0]

        indices = np.arange(len(input_blocks)).reshape(*prev_layer.output_dims)
        for output_channel in range(n_output_channels): # no of output channels is most outer loop
            patches = image.extract_patches_2d(indices[output_channel,:,:], (kernel_size)) # extract patches with stride 1
            patches = np.stack(patches)
            patches_side_length = int(np.sqrt(patches.shape[0]))
            patches = patches.reshape(patches_side_length, patches_side_length, *kernel_size, -1) # align patches as a rectangle
            # pick only patches that are interesting (stride)
            patches = patches[::self.stride,::self.stride,:,:,:].reshape(-1, *kernel_size, patches.shape[-1])
            for i in range(int(np.product(self.output_dims[1:3]))): # loop through all units in the output channel
                maxpool = quartz.blocks.WTA(name="pool-c{1:3.0f}-n{2:3.0f}:".format(self.layer_n, output_channel, i), parent_layer=self)
                block_patch = np.array(input_blocks)[patches[i,:,:,:].flatten()]
                n_inputs = len(block_patch)
                for block in block_patch:
                    acc1 = Neuron(name=maxpool.name + "acc1_{}".format(i), loihi_type=Neuron.acc, parent=maxpool)
                    acc2 = Neuron(name=maxpool.name + "acc2_{}".format(i), loihi_type=Neuron.acc, parent=maxpool)
                    block.first().connect_to(acc1, self.weight_acc)
                    block.second().connect_to(acc2, self.weight_acc)
                    acc1.connect_to(acc2, -self.weight_acc)
                    acc1.connect_to(maxpool.neurons[0], self.weight_e/n_inputs)
                    acc1.connect_to(acc1, -self.weight_acc)
                    acc2.connect_to(maxpool.neurons[2], self.weight_e/n_inputs)
                    acc2.connect_to(acc2, -self.weight_acc)
                    maxpool.neurons[0].connect_to(acc2, self.weight_acc)
                    maxpool.neurons += [acc1, acc2]
                self.blocks += [maxpool]


class MonitorLayer(Layer):
    def __init__(self, name="monitor:", **kwargs):
        super(MonitorLayer, self).__init__(name=name, **kwargs)

    def connect_from(self, prev_layer):
        self.layer_n = prev_layer.layer_n + 1
        self.prev_layer = prev_layer
        self.name = "l{}-{}".format(self.layer_n, self.name)
        self.output_dims = prev_layer.output_dims
        
        for i, block in enumerate(prev_layer.output_blocks()):
            monitor = quartz.blocks.Block(name=block.name+"monitor", type=Block.output, monitor=self.monitor, parent_layer=self)
            output_neuron = Neuron(name=block.name + "monitor-{0:3.0f}".format(i), type=Block.input, monitor=self.monitor, parent=monitor)
            monitor.neurons += [output_neuron]
            self.blocks += [monitor]
            extra_second_delay = 2 if isinstance(block, quartz.blocks.ConvMax) else 0
            block.first().connect_to(output_neuron, self.weight_e)
            block.second().connect_to(output_neuron, self.weight_e, extra_second_delay)
