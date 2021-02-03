import quartz
import numpy as np


class Block:
    def __init__(self, neurons, name=None, monitor=False):
        self.neurons = neurons
        self.name = name
        self.monitor = monitor
        self.connections = []

    def connect_to(self, target, weight, exponent=0, delay=0):
        self.connections.append((target, weight, exponent, delay))
        for n, neuron in enumerate(self.neurons):
            neuron.n_outgoing_synapses += np.sum(weight[:,n] != 0)
        if isinstance(target, quartz.components.Neuron):
            target.n_incoming_synapses += np.sum(weight != 0)
        else: # connect to other block
            for n, neuron in enumerate(target.neurons):
                neuron.n_incoming_synapses += np.sum(weight[n,:] != 0)

    def __repr__(self):
        return self.name


class Neuron:
    input, output, bias, sync, rectifier = range(5)
    pulse, acc = range(2)

    def __init__(self, type=output, loihi_type=pulse, name=None, monitor=False):
        self.type = type
        self.loihi_type = loihi_type
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
