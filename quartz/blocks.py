import numpy as np
from quartz.components import Neuron
import quartz
import ipdb
from collections import Counter


class Block:
    input, hidden, output, trigger = range(4)
    
    def __init__(self, name, parent_layer, type=hidden, monitor=False):
        self.name = name
        self.type = type
        self.monitor = monitor
        self.neurons = []
        self.parent_layer = parent_layer
        self.input_neurons = []
        self.output_neurons = []
        self.rectifier_neurons = []
    
    def neurons(self): return self.neurons
    
    def names(self): return [neuron.name for neuron in self.neurons]

    def neuron(self, name):
        res = tuple(neuron for neuron in self.neurons if name in neuron.name)
        if len(res) > 1:
            [print(neuron.name) for neuron in res]
            raise Exception("Provided name was ambiguous, results returned: ")
        elif len(res) == 0:
            raise Exception("No neuron with name {} found. Available names are: {}".format(name, self.neurons))
        return res[0]
    
    def first(self):
        return self.output_neurons[0]

    def second(self):
        return self.output_neurons[1]

    def print_connections(self, maximum=10e7):
        for i, neuron in enumerate(self.neurons):
            if neuron.synapses != []: 
                [print(connection) for connection in neuron.synapses]
            elif neuron.type != Neuron.output and neuron.type != Neuron.ready:
                print("Warning: neuron {} seems not to be connected to any other neuron.".format(neuron.name))
            if i > maximum: break

    def n_compartments(self):
        return len(self.neurons)

    def n_outgoing_connections(self):
        n_conns = 0
        for neuron in self.neurons:
            for synapse in neuron.synapses:
                if synapse[0].parent_block != synapse[1].parent_block: n_conns += 1
        return n_conns
    
    def n_recurrent_connections(self):
        n_conns = 0
        for neuron in self.neurons:
            for synapse in neuron.synapses:
                if synapse[0].parent_block == synapse[1].parent_block: n_conns += 1
        return n_conns

    def get_params_at_once(self):
        return self.parent_layer.get_params_at_once()
    
    def get_connected_blocks(self):
        connected_blocks = []
        for neuron in self.neurons:
            for synapse in neuron.synapses:
                connected_blocks.append(synapse[1].parent_block)
        unique_blocks = []
        for block in connected_blocks:
            if block not in unique_blocks:
                unique_blocks.append(block)
        return unique_blocks

    def get_connection_matrices_to(self, block):
        ok = block
        all_synapses = [synapse for neuron in self.neurons for synapse in neuron.synapses]
        relevant_synapses = [(pre, post, weight, delay) for pre, post, weight, delay in all_synapses if (post in block.neurons) & (weight != 0)]
        endpoints = [(pre, post) for pre, post, weight, delay in relevant_synapses]
        counter=Counter(endpoints)
        if relevant_synapses == []: return None, None, None, None
        
        max_n_conn_between_endpoints = max(counter.values())
        pre_index_list = dict(zip(self.neurons, range(len(self.neurons))))
        post_index_list = dict(zip(block.neurons, range(len(block.neurons))))
        
        weights = np.zeros((max_n_conn_between_endpoints, len(block.neurons), len(self.neurons)))
        delays = np.zeros_like(weights)
        exponents = np.zeros_like(weights)
        
        c = 0
        factors = [64, 32, 16, 8, 4, 2]
        factor_exponents = [6, 5, 4, 3, 2, 1]
        for pre, post, weight, delay in relevant_synapses:
            i = pre_index_list[pre]
            j = post_index_list[post]
            if weights[c,j,i] != 0: c+=1
            delays[c,j,i] = delay
            if abs(weight) > 255:
                for e, factor in enumerate(factors):
                    if (weight / factor).is_integer() & (-256 < (weight / factor) < 256):
                        weights[c,j,i] = weight / factor
                        exponents[c,j,i] = factor_exponents[e]
                        break
            else:
                weights[c,j,i] = weight
                exponents[c,j,i] = 0
        mask = np.full(weights.shape, False)
        mask[weights!=0] = True
        return weights, delays, exponents, mask
    
    def __repr__(self):
        return self.name


class Input(Block):
    def __init__(self, name="input:", type=Block.output, **kwargs):
        super(Input, self).__init__(name=name, type=type, **kwargs)
        output = Neuron(type=Neuron.output, name=self.name+"neuron", parent=self)
        self.neurons = [output]
        self.output_neurons += [output]
        self.rectifier_neurons += [output]
    
    
class Bias(Block):
    def __init__(self, value, name="bias:", type=Block.hidden, **kwargs):
        self.value = abs(value)
        super(Bias, self).__init__(name=name, type=type, **kwargs)
        input_ = Neuron(name=self.name+"input", parent=self)
        output = Neuron(name=self.name+"output", parent=self)
        self.neurons = [input_, output]
        self.input_neurons += [input_]
        self.output_neurons += [output]
        self.layout = False

    def layout_delays(self, t_max, numDendriticAccumulators):
        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        # subtract 2*t_neu for delay of input, output neurons
        delay = np.maximum((t_max*(1-self.value)).round() - 2*t_neu, 0)
        numDendriticAccumulators = numDendriticAccumulators-2
        input_, output = self.neurons
        self.neurons = [input_]
        i = 0
        while(delay>(numDendriticAccumulators)):
            intermediate = Neuron(name=self.name+"intermediate"+str(i), parent=self)
            self.neurons[-1].connect_to(intermediate, weight_e, numDendriticAccumulators)
            self.neurons += [intermediate]
            delay -= (numDendriticAccumulators + 1)
            i += 1
        self.neurons[-1].connect_to(output, weight_e, delay)
        self.neurons += [output]
        self.layout = True


class ReLCo(Block): # Rectifying Linear Combination
    def __init__(self, name="relco:", type=Block.output, **kwargs):
        super(ReLCo, self).__init__(name=name, type=type, **kwargs)
        calc = Neuron(name=name + "calc", loihi_type=Neuron.acc, parent=self)
        self.neurons = [calc]
        self.input_neurons += [calc]
        self.output_neurons += [calc]
        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        calc.connect_to(calc, -2**6*255)

        
class WTA(Block):
    def __init__(self, name="maxpool:", type=Block.output, **kwargs):
        super(WTA, self).__init__(name=name, type=type, **kwargs)
        first = Neuron(name=name + "1st", loihi_type=Neuron.pulse, parent=self)
        self.neurons = [first]
        self.input_neurons += [first]
        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        first.connect_to(first, -8.1*weight_e)


class Trigger(Block):
    def __init__(self, n_channels, name="trigger:", type=Block.trigger, **kwargs):
        super(Trigger, self).__init__(name=name, type=type, **kwargs)
        rect = Neuron(name=name + "rect", loihi_type=Neuron.acc, type=Neuron.rectifier, parent=self)
        self.neurons = [rect]
        self.rectifier_neurons += [rect]
        assert n_channels > 0
        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        rect.connect_to(rect, -weight_acc)
        for t in range(n_channels):
            trigger_neuron = Neuron(name=self.name + "trigger" + str(t), type=Neuron.output, parent=self)
            self.neurons += [trigger_neuron]
            self.output_neurons += [trigger_neuron]
