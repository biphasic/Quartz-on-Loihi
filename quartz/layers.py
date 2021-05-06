from quartz.components import Block, Neuron
import quartz
from sklearn.feature_extraction import image
import numpy as np


class Layer:
    """
    Parent class for all layers in the model, before being compiled to a backend. The connections can happen between individual neurons in the case of single weights, or blocks of neurons in the case of weight matrices.
    
    Args:
        name: layer name. 
        weight_acc: should be set to the maximum weight resolution of the backend. Higher number gives more fine-grained weight resolution.
        t_min: the number of time steps that encode the value of zero. Since this framework uses inter spike intervals, even zeros have to be encoded (for the inputs).
        t_neu: the number of time steps it takes for a neuron to spike. 
        monitor: flag that decides whether all neurons in the layer are probed. Usually set by quartz.probe
        
    Returns:
        layer object which will be automatically connected to previous layer and performs automatic checks on input/output dimensions.
    """
    def __init__(self, name, weight_acc=254, t_min=1, t_neu=1, monitor=False):
        self.name = name
        self.weight_e = 30
        self.weight_acc = weight_acc
        self.t_min = t_min
        self.t_neu = t_neu
        self.monitor = monitor
        self.output_dims = []
        self.layer_n = None
        self.blocks = []
        self.output_neurons = []
        self.bias_neurons = []
        self.sync_neurons = []
        self.rectifier_neurons = []
        self.num_dendritic_accumulators = 2**3
        self.max_axonal_delay = 63*3

    def neurons(self): return self.output_neurons + self.sync_neurons + self.rectifier_neurons + self.bias_neurons
    
    def neurons_without_bias(self): return self.output_neurons + self.sync_neurons + self.rectifier_neurons

    def names(self): return [neuron.name for neuron in self.neurons()]
    
    def n_output_compartments(self):
        if isinstance(self, quartz.layers.InputLayer): return 0
        return len(self.neurons_without_bias())
    
    def n_bias_compartments(self):
        if isinstance(self, quartz.layers.InputLayer): return 0
        return len(self.bias_neurons)

    def n_parameters(self):
        if isinstance(self, (quartz.layers.MaxPool2D, quartz.layers.InputLayer)): return 0
        n_params = np.product(self.weights.shape)
        if self.biases is not None: n_params += np.product(self.biases.shape)
        return n_params

    def n_outgoing_connections(self):
        return sum([np.sum(connection[1]!=0) for block in self.blocks for connection in block.connections])\
                + sum([len(neuron.synapses) for neuron in self.neurons()])

    def __repr__(self):
        return self.name


class InputLayer(Layer):
    """
    Input layer is needed so that neuron can be tiled properly when connected to the first layer. When compiled to the backend, this layer does not need any neurons/compartments, but uses input generators instead.
    
    Args:
        dims: input dimensions.
        name: layer name. 
        
    Returns:
        input layer object.
    """
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
    """
    A linear layer with optional relu activation. Weights and biases have to be passed from the model that is being converted. 
    
    Args:
        weights: the actual weights with dimensions (OUTxINPUT).
        biases: optional biases with dimensions (OUT).
        rectifying: whether to use relu activation or not.
        name: layer name. 
        
    Returns:
        dense layer object.
    """
    def __init__(self, weights, biases=None, rectifying=True, name="dense:", **kwargs):
        super(Dense, self).__init__(name=name, **kwargs)
        self.weights = weights.copy()
        self.biases = biases
        self.rectifying = rectifying
        self.output_dims = weights.shape[0]

    def __repr__(self):
        s = "{} features_in={}, features_out={}".format(self.name, self.weights.shape[0], self.weights.shape[1])
        if self.biases is not None:
            s += ', bias=True'
        if self.rectifying:
            s += ', relu=True'
        return s

    def connect_from(self, prev_layer, t_max):
        self.layer_n = prev_layer.layer_n + 1
        self.name = "l{}-{}".format(self.layer_n, self.name)
        weights, biases = self.weights, self.biases
        weights = np.ceil(weights*self.weight_acc)
        weights[weights%2==1] -= 1 # 7 bit weight precision on Loihi with mixed weight coefficients
        weights /= self.weight_acc
        assert weights.shape[1] == len(prev_layer.output_neurons)
        if biases is not None: assert weights.shape[0] == biases.shape[0]

        # create and connect support neurons
        self.sync_neurons = [Neuron(name=self.name+"sync:", type=Neuron.sync)]
        self.rectifier_neurons = [Neuron(name=self.name+"rectifier:", type=Neuron.rectifier, current_type=Neuron.accumulation)]
        self.rectifier_neurons[0].connect_to(self.rectifier_neurons[0], -self.weight_acc)
        prev_layer.rectifier_neurons[0].connect_to(self.sync_neurons[0], self.weight_e)
        prev_layer.rectifier_neurons[0].connect_to(self.rectifier_neurons[0], self.weight_acc)
        
        # create all output neurons and create self-inhibiting group connection
        self.output_neurons = [Neuron(name=self.name+"relco-n{0:3.0f}:".format(i), current_type=Neuron.accumulation) for i in range(self.weights.shape[0])]
        layer_neuron_block = Block(neurons=self.output_neurons, name=self.name+"all-units")
        layer_neuron_block.connect_to(layer_neuron_block, -255*np.eye(len(self.output_neurons)), 6, 0)
        self.blocks += [layer_neuron_block]

        # group neurons from previous layer
        output_block = Block(neurons=prev_layer.output_neurons, name=prev_layer.name+"dense-block")
        prev_layer.blocks += [output_block]
        # connections between layers
        output_block.connect_to(layer_neuron_block, weights*self.weight_acc, 0, 0)
        
        # balancing connections from sync neuron for this layer
        sync_block = Block(neurons=self.sync_neurons, name=self.name+"sync-block")
        if biases is None: biases = np.zeros((self.output_dims))
        weight_sums = np.array([(1 - sum(weights[output,:]) - np.sign(biases[output]))*self.weight_acc for output in range(self.output_dims)]).round().astype(int)
        # most significant bits of counter weight minus 1 because 7 bit weight precision
        weight_sum_base = np.left_shift(np.right_shift(abs(weight_sums), 7), 1) * np.sign(weight_sums) 
        # least 7 significant bits (6 for weight exponent + 1 lost one)
        weight_sum_precision = np.bitwise_and(abs(weight_sums), 0b1111111) * np.sign(weight_sums) 
        assert (weight_sum_base * 2**6 + weight_sum_precision == weight_sums).all()
        sync_block.connect_to(layer_neuron_block, np.array(weight_sum_base).reshape(len(weight_sums), len(self.sync_neurons)), 6, 0)
        sync_block.connect_to(layer_neuron_block, np.array(weight_sum_precision).reshape(len(weight_sum_precision), len(self.sync_neurons)), 0, 0)
        self.blocks += [sync_block]

        # connect biases
        for b, bias in enumerate(biases):
            if bias != 0:
                bias_sign = np.sign(bias)
                delay = np.maximum(round((1-abs(bias))*t_max) - 1, 0)
                self.bias_neurons += [Neuron(name=self.name+"bias-{}:".format(b), type=Neuron.bias)]
                source = self.bias_neurons[-1]
                prev_layer.sync_neurons[0].connect_to(source, self.weight_e)
                while delay > (self.max_axonal_delay):
                    self.bias_neurons += [Neuron(name=self.name+"bias-{}:".format(b), type=Neuron.bias)]
                    source.connect_to(self.bias_neurons[-1], self.weight_e, 0, self.max_axonal_delay)
                    source = self.bias_neurons[-1]
                    delay -= self.max_axonal_delay+self.t_neu
                source.connect_to(self.output_neurons[b], bias_sign*self.weight_acc, 0, delay)

        if self.rectifying:
            rectifier_block = Block(neurons=self.rectifier_neurons, name=self.name+"rectifier-block")
            rectifier_block.connect_to(layer_neuron_block, np.array([251]*len(self.output_neurons)).reshape(len(self.output_neurons),len(self.rectifier_neurons)), 6, 0)
            self.blocks += [rectifier_block]


class Conv2D(Layer):
    """
    A convolutional layer with optional relu activation. Weights and biases have to be passed from the model that is being converted. Supports stride, padding, groups. Modeled after [pytorch conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) layer. 
    
    Args:
        weights: the actual weights with dimensions (OUTxINPUT).
        biases: optional biases with dimensions (OUT).
        stride: controls the stride for the cross-correlation, a single number or a tuple.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points for each dimension.
        groups: controls the connections between inputs and outputs. Can be used for depthwise separable convolutions by setting this parameter to the number of input channels.
        rectifying: whether to use relu activation or not.
        name: layer name. 
        
    Returns:
        conv layer object.
    """
    def __init__(self, weights, biases=None, stride=(1,1), padding=(0,0), groups=1, rectifying=True, name="conv2D:", monitor=False, **kwargs):
        super(Conv2D, self).__init__(name=name, monitor=monitor, **kwargs)
        self.weights = weights.copy()
        self.biases = biases
        if isinstance(stride, int): stride = (stride, stride)
        self.stride = stride
        if isinstance(padding, int): padding = (padding, padding)
        self.padding = padding
        self.groups = groups
        self.rectifying = rectifying
            
    def __repr__(self):
        s = "{} {}, {}, kernel_size={}".format(self.name, self.weights.shape[0], self.weights.shape[1], self.weights.shape[2:])
        if self.padding != (0,) * len(self.padding):
            s += ', padding={}'.format(self.padding)
        if self.groups != 1:
            s += ', groups={}'.format(self.groups)
        if self.biases is None:
            s += ', bias=False'
        if self.rectifying:
            s += ', relu=True'
        return s

    def connect_from(self, prev_layer, t_max):
        self.prev_layer = prev_layer
        self.layer_n = prev_layer.layer_n + 1
        self.name = "l{}-{}".format(self.layer_n, self.name)
        weights, biases = self.weights, self.biases
        weights = np.ceil(weights*self.weight_acc)
        weights[weights%2==1] -= 1 # 7 bit weight precision on Loihi with mixed weight coefficients
        weights /= self.weight_acc
        assert weights.shape[1]*self.groups == prev_layer.output_dims[0]
        if biases is not None: assert weights.shape[0] == biases.shape[0]
        output_channels, input_channels, *kernel_size = self.weights.shape
        input_channels *= self.groups
        side_lengths = (int((prev_layer.output_dims[1] - kernel_size[0] + 2*self.padding[0]) / self.stride[0] + 1), 
                        int((prev_layer.output_dims[2] - kernel_size[1] + 2*self.padding[1]) / self.stride[1] + 1))
        self.output_dims = (output_channels, *side_lengths)
        
        # create and connect support neurons
        self.sync_neurons = [Neuron(name=self.name+"sync:", type=Neuron.sync)]
        self.rectifier_neurons = [Neuron(name=self.name+"rectifier:", type=Neuron.rectifier, current_type=Neuron.accumulation)]
        self.rectifier_neurons[0].connect_to(self.rectifier_neurons[0], -self.weight_acc)
        prev_layer.rectifier_neurons[0].connect_to(self.sync_neurons[0], self.weight_e)
        prev_layer.rectifier_neurons[0].connect_to(self.rectifier_neurons[0], self.weight_acc)
        
        # create all output neurons 
        self.output_neurons = [Neuron(name=self.name+"relco-c{0:3.0f}-n{1:3.0f}:".format(output_channel, i), current_type=Neuron.accumulation) 
                               for output_channel in range(output_channels) for i in range(np.product(side_lengths))]

        input_neurons = prev_layer.output_neurons
        input_neurons = np.pad(np.array(input_neurons).reshape(*prev_layer.output_dims), 
                              ((0,0), self.padding, self.padding), 'constant', constant_values=(0))
        indices = np.arange(len(input_neurons.ravel())).reshape(input_neurons.shape)
        input_neurons = input_neurons.ravel()
        n_groups_out = output_channels/self.groups
        n_groups_in = input_channels//self.groups
        assert isinstance(n_groups_out, int) or n_groups_out.is_integer() # and n_groups_in.is_integer()
        n_groups_out = int(n_groups_out)

        weight_sums = np.zeros((output_channels, np.product(side_lengths),))
        if biases is None: biases = np.zeros((output_channels))
        for g in range(self.groups): # split feature maps into groups
            for output_channel in range(g*n_groups_out,(g+1)*n_groups_out): # loop over output channels in one group
                patches = [image.extract_patches_2d(indices[input_channel,:,:], (kernel_size)) for input_channel in range(input_channels)]
                patches = np.stack(patches)
                # stride patches
                patches_side_length = int(np.sqrt(patches.shape[1]))
                patches = patches.reshape(-1, patches_side_length, patches_side_length, *kernel_size) # align patches as a rectangle
                patches = patches[:, ::self.stride[0],::self.stride[1],:,:].reshape(patches.shape[0], -1, *kernel_size) 
                assert np.product(side_lengths) == patches.shape[1]
                
                # create bias per output channel
                bias = biases[output_channel]
                if bias != 0:
                    bias_sign = np.sign(bias)
                    delay = np.maximum(round((1-abs(bias))*t_max) - 1, 0)
                    self.bias_neurons += [Neuron(name=self.name+"bias-{}:".format(output_channel), type=Neuron.bias)]
                    source = self.bias_neurons[-1]
                    prev_layer.sync_neurons[0].connect_to(source, self.weight_e)
                    while delay > (self.max_axonal_delay):
                        self.bias_neurons += [Neuron(name=self.name+"bias-{}:".format(output_channel), type=Neuron.bias)]
                        source.connect_to(self.bias_neurons[-1], self.weight_e, 0, self.max_axonal_delay)
                        source = self.bias_neurons[-1]
                        delay -= self.max_axonal_delay+self.t_neu
                    weight_sums[output_channel, :] -= bias_sign
                
                for i in range(np.product(side_lengths)): # loop through all units in the output channel
                    # connect bias to every neuron in output channel
                    if bias != 0: source.connect_to(self.output_neurons[output_channel*np.product(side_lengths)+i], bias_sign*self.weight_acc, 0, delay)

                    # connect output neurons from previous layer
                    for group_weight_index, input_channel in enumerate(range(g*n_groups_in,(g+1)*n_groups_in)):
                        receptive_field = input_neurons[patches[input_channel,i,:,:].ravel()]
                        mask = receptive_field != 0
                        block_patch = Block(neurons=list(receptive_field[mask]), name=prev_layer.name+"patch-c{0:3.0f}-n{1:3.0f}:".format(input_channel, i))
                        prev_layer.blocks += [block_patch]
                        patch_weights = weights[output_channel,group_weight_index,:,:].ravel()
                        patch_weight_selection = patch_weights[mask]
                        assert len(block_patch.neurons) == patch_weight_selection.shape[0]
                        block_patch.connect_to(self.output_neurons[output_channel*np.product(side_lengths)+i], 
                                               patch_weight_selection.reshape(1, len(patch_weight_selection))*self.weight_acc)
                        weight_sums[output_channel, i] -= sum(patch_weight_selection)

        # recurring self-inhibitory connection
        layer_neuron_block = Block(neurons=self.output_neurons, name=self.name+"all-units")
        layer_neuron_block.connect_to(layer_neuron_block, -255*np.eye(len(self.output_neurons)), 6, 0)
        self.blocks += [layer_neuron_block]

        # connect sync counter weights
        sync_block = Block(neurons=self.sync_neurons, name=self.name+"sync-block")
        if biases is None: biases = np.zeros((self.output_dims))
        weight_sums = ((weight_sums.flatten() + 1) * self.weight_acc).round().astype(int)
        # most significant bits of counter weight minus 1 because 7 bit weight precision
        weight_sum_base = np.left_shift(np.right_shift(abs(weight_sums), 7), 1) * np.sign(weight_sums) 
        # least 7 significant bits (6 for weight exponent + 1 lost one)
        weight_sum_precision = np.bitwise_and(abs(weight_sums), 0b1111111) * np.sign(weight_sums) 
        assert (weight_sum_base * 2**6 + weight_sum_precision == weight_sums).all()
        sync_block.connect_to(layer_neuron_block, np.array(weight_sum_base).reshape(len(weight_sums), len(self.sync_neurons)), 6, 0)
        sync_block.connect_to(layer_neuron_block, np.array(weight_sum_precision).reshape(len(weight_sum_precision), len(self.sync_neurons)), 0, 0)
        self.blocks += [sync_block]

        if self.rectifying:
            rectifier_block = Block(neurons=self.rectifier_neurons, name=self.name+"rectifier-block")
            rectifier_block.connect_to(layer_neuron_block, np.array([251]*len(self.output_neurons)).reshape(len(self.output_neurons),len(self.rectifier_neurons)), 6, 0)
            self.blocks += [rectifier_block]


class MaxPool2D(Layer):
    """
    A maxpooling layer. Kernel size has to be passed from the model that is being converted. 
    
    Args:
        kernel_size: size of the pooling kernel
        stride: controls the stride of the pooling kernel, a single number or a tuple.
        name: layer name. 
        
    Returns:
        maxpooling layer object.
    """
    def __init__(self, kernel_size, stride=None, name="maxpool:", **kwargs):
        super(MaxPool2D, self).__init__(name=name, **kwargs)
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
            
    def __repr__(self):
        return "{} kernel_size={}, stride={}".format(self.name, self.kernel_size, self.stride)

    def connect_from(self, prev_layer, t_max):
        self.prev_layer = prev_layer
        self.layer_n = prev_layer.layer_n + 1
        self.name = "l{}-{}".format(self.layer_n, self.name)
        self.output_dims = list(prev_layer.output_dims)
        output_channels = self.output_dims[0]
        if self.stride==None: self.stride = self.kernel_size[0]
        self.output_dims[1] = int(self.output_dims[1]/self.kernel_size[0])
        self.output_dims[2] = int(self.output_dims[2]/self.kernel_size[1])
        
        # create and connect support neurons
        self.sync_neurons = [Neuron(name=self.name+"sync:", type=Neuron.sync)]
        self.rectifier_neurons = [Neuron(name=self.name+"rectifier:", type=Neuron.rectifier)]
        prev_layer.sync_neurons[0].connect_to(self.sync_neurons[0], self.weight_e)
        prev_layer.rectifier_neurons[0].connect_to(self.rectifier_neurons[0], self.weight_e)
        
        # create output neurons 
        self.output_neurons += [Neuron(name=self.name+"wta-c{0:3.0f}-n{1:3.0f}:".format(output_channel, i), current_type=Neuron.instant)
                               for output_channel in range(output_channels) for i in range(np.product(self.output_dims[1:]))]

        input_neurons = np.array(prev_layer.output_neurons)
        indices = np.arange(len(input_neurons)).reshape(prev_layer.output_dims)
        for output_channel in range(output_channels): # no of output channels is most outer loop
            patches = image.extract_patches_2d(indices[output_channel,:,:], (self.kernel_size)) # extract patches with stride 1
            patches = np.stack(patches)
            patches_side_length = int(np.sqrt(patches.shape[0]))
            patches = patches.reshape(patches_side_length, patches_side_length, *self.kernel_size, -1) # align patches as a rectangle
            # stride patches
            patches = patches[::self.stride,::self.stride,:,:,:].reshape(-1, *self.kernel_size, patches.shape[-1]) 
            for i in range(int(np.product(self.output_dims[1:]))): # loop through all units in the output channel
                block_patch = Block(neurons=list(input_neurons[patches[i,:,:,:].ravel()]))
                prev_layer.blocks += [block_patch]
                block_patch.connect_to(self.output_neurons[output_channel*np.product(self.output_dims[1:])+i], 
                                       np.array([self.weight_e]*len(block_patch.neurons)).reshape(1, len(block_patch.neurons)))

        # recurring self-inhibitory connection
        layer_neuron_block = Block(neurons=self.output_neurons, name=self.name+"all-units")
        layer_neuron_block.connect_to(layer_neuron_block, -255*np.eye(len(self.output_neurons)), 6)
        self.blocks += [layer_neuron_block]