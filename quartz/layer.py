import numpy as np
import quartz
from quartz.components import Neuron, Synapse


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

    def output_neurons(self): return self._get_neurons_of_type(Neuron.output)

#     def output_neurons(self):
#         return [neuron for neuron in self.neurons if neuron.type == Neuron.output]
#         neurons = [block.output_neurons() for block in self.blocks if isinstance(block, quartz.blocks.ReLCo)]
#         return [neuron for pair in neurons for neuron in pair]
    
    def get_params_at_once(self):
        return self.weight_e, self.weight_acc, self.t_syn, self.t_min, self.t_neu
    
    
    def __repr__(self):
        return self.name