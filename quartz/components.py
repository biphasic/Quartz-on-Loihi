import quartz
import numpy as np


class Block:
    """
    A block defines a group of neurons. A block can connect to another block or a single neuron, using weight, delay and weight exponent matrices. 
    Will keep track of number of incoming and outgoing synapses. 
    
    Args:
        neurons: list of neurons that belong to the block. flattened dimensions should be equal to weight/delay matrix
        name: layer name. 
        monitor: flag that decides whether all neurons in the block are probed. Usually set by quartz.probe
        
    Returns:
        block object that can define connections in a fast and memory efficient way.
    """
    def __init__(self, neurons, name=None, monitor=False):
        self.neurons = neurons
        self.name = name
        self.monitor = monitor
        self.connections = []

    def connect_to(self, target, weight, exponent=0, delay=0):
        self.connections.append((target, weight, exponent, delay))
#         for n, neuron in enumerate(self.neurons):
#             neuron.n_outgoing_synapses += np.sum(weight[:,n] != 0)
#         if isinstance(target, quartz.components.Neuron):
#             target.n_incoming_synapses += np.sum(weight != 0)
#         else: # connect to other block
#             for n, neuron in enumerate(target.neurons):
#                 neuron.n_incoming_synapses += np.sum(weight[n,:] != 0)

    def __repr__(self):
        return self.name


class Neuron:
    """
    A neuron is a unit that has been converted from the ANN model. A neuron can be of instant input current decay or accumulation (no current decay) type and can connect to another neuron or a block.
    
    Args:
        type: can be input, output, bias, sync or rectifier. Little significance for the model, but interesting when compiling to backend.
        current_type: can be instant or accumulating. Equivalent to V-neuron or g_e neuron in the paper.  
        monitor: flag that decides whether neuron is probed. Usually set by quartz.probe
        
    Returns:
        neuron object
    """
    input, output, bias, sync, rectifier = range(5)
    instant, accumulation = range(2)

    def __init__(self, type=output, current_type=instant, name=None, monitor=False):
        self.type = type
        self.current_type = current_type
        self.loihi_block = None
        self.name = name
        self.monitor = monitor
        self.synapses = []
        self.n_incoming_synapses = 0
        self.n_outgoing_synapses = 0

    def connect_to(self, target_neuron, weight, exponent=0, delay=0):
        self.synapses.append((target_neuron, weight, exponent, delay))
        self.n_outgoing_synapses += 1
        target_neuron.n_incoming_synapses += 1

    def __repr__(self):
        return self.name
