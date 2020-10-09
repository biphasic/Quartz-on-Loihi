import numpy as np
from quartz.components import Neuron, Synapse
import quartz
import ipdb
 

class Block:
    input, hidden, output, trigger = range(4)
    
    def __init__(self, name, parent_layer, type=hidden, monitor=False):
        self.name = name
        self.type = type
        self.monitor = monitor
        self.neurons = []
        self.parent_layer = parent_layer
        self.connections = {"pre": [], "post": []}
    
    def neurons(self): return self.neurons
    
    def names(self): return [neuron.name for neuron in self.neurons]

    def _get_neurons_of_type(self, neuron_type):
        return [neuron for neuron in self.neurons if neuron.type == neuron_type]
    
    def input_neurons(self): return self._get_neurons_of_type(Neuron.input)

    def output_neurons(self): return self._get_neurons_of_type(Neuron.output)

    def first(self): 
        neuron = self.output_neurons()[0]
        assert "1st" in neuron.name
        return neuron

    def second(self):
        neuron = self.output_neurons()[1]
        assert "2nd" in neuron.name
        return neuron

    def print_connections(self, maximum=10e7):
        for i, neuron in enumerate(self.neurons):
            if neuron.synapses['pre'] != []: 
                [print(connection) for connection in neuron.synapses['pre']]
            elif neuron.type != Neuron.output and neuron.type != Neuron.ready:
                print("Warning: neuron {} seems not to be connected to any other neuron.".format(neuron.name))
            if i > maximum: break

    def n_compartments(self):
        return len(self.neurons)

    def n_connections(self):
        return sum([len(neuron.synapses['post']) for neuron in self.neurons])

    def get_params_at_once(self):
        return self.parent_layer.get_params_at_once()
    
    def get_connected_blocks(self):
        connected_blocks = []
        for neuron in self.neurons:
            for synapse in neuron.incoming_synapses():
                connected_blocks.append(synapse.pre.parent_block)
        unique_blocks = []
        for block in connected_blocks:
            if block not in unique_blocks:
                unique_blocks.append(block)
        return unique_blocks # set(connected_blocks)

    def incoming_connections(self):
        return self.connections["post"]

    def outgoing_connections(self):
        return self.connections["pre"]

    def has_incoming_connections(self):
        return self.connections["post"] != []

    def get_connection_matrices_to(self, block):
        weights = np.zeros((len(block.neurons), len(self.neurons)))
        delays = np.zeros_like(weights)
        mask = np.zeros_like(weights)
        for i, neuron in enumerate(self.neurons):
            for synapse in neuron.outgoing_synapses():
                if synapse.post in block.neurons:
                    j = block.neurons.index(synapse.post)
                    weights[j, i] = synapse.weight
                    delays[j, i] = synapse.delay
        mask[weights!=0] = 1
        return weights, delays, mask


class ConstantDelay(Block):
    def __init__(self, value, name="bias:", type=Block.hidden, **kwargs):
        self.value = abs(value)
        super(ConstantDelay, self).__init__(name=name, type=type, **kwargs)
        input_ = Neuron(type=Neuron.input, name=self.name+"input", parent=self)
        output = Neuron(type=Neuron.output, name=self.name+"output", parent=self)
        self.neurons = [input_, output]            
        self.reset()

    def reset(self):
        if len(self.neurons) > 2:
            input_ = self.neurons[0]
            input_.reset_outgoing_connections()
            output = self.neurons[-1]
            output.reset_incoming_connections()
            self.neurons = [input_, output]            
        else:
            pass
#         for neuron in self.neurons:
#             neuron.reset_outgoing_connections()
#         input_ = Neuron(type=Neuron.input, name=self.name+"input", parent=self)
#         output = Neuron(type=Neuron.output, name=self.name+"output", parent=self)
#         self.neurons = [input_, output]

    def layout_delays(self, t_max, numDendriticAccumulators):
        delay = round(self.value * t_max)
        numDendriticAccumulators = numDendriticAccumulators-2
        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        input_, output = self.neurons
        self.neurons = [] + [input_]
        i = 0
        while(delay>(numDendriticAccumulators-t_min)):
            intermediate = Neuron(name=self.name+"intermediate"+str(i), parent=self)
            self.neurons[-1].connect_to(intermediate, weight_e, numDendriticAccumulators)
            self.neurons += [intermediate]
            delay -= numDendriticAccumulators
            i += 1
        self.neurons[-1].connect_to(output, weight_e, delay+t_min)

        delay = i*t_neu
        self.neurons.append(self.neurons.pop(0)) # move input_ to the end
        i = 0
        while(delay>numDendriticAccumulators):
            intermediate = Neuron(name=self.name+"intermediate-output"+str(i), parent=self)
            self.neurons[-1].connect_to(intermediate, weight_e, numDendriticAccumulators)
            self.neurons += [intermediate]
            delay -= (numDendriticAccumulators+1)
            i += 1
        self.neurons[-1].connect_to(output, weight_e, delay)
        self.neurons += [output]


class Splitter(Block):
    def __init__(self, name="split:", type=Block.hidden, **kwargs):
        super(Splitter, self).__init__(name=name, type=type, **kwargs)
        input_ = Neuron(type=Neuron.input, name=name + "input", parent=self)
        first = Neuron(type=Neuron.output, name=name + "1st", parent=self)
        last = Neuron(type=Neuron.output, name=name + "2nd", parent=self)
        self.neurons = [input_, first, last]
        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        input_.connect_to(first, weight_e)
        first.connect_to(first, -weight_e)
        input_.connect_to(last, 0.5*weight_e)


class ReLCo(Block):
    def __init__(self, name="relco:", type=Block.output, **kwargs):
        super(ReLCo, self).__init__(name=name, type=type, **kwargs)
        calc = Neuron(name=name + "calc", loihi_type=Neuron.acc, type=Neuron.input, parent=self)
        ref = Neuron(name=name + "ref", loihi_type=Neuron.acc, parent=self)
        sync = Neuron(name=name + "sync", type=Neuron.input, parent=self)
        first = Neuron(name=name + "1st", type=Neuron.output, parent=self)
        second = Neuron(name=name + "2nd", type=Neuron.output, parent=self)
        self.neurons = [calc, ref, sync, first, second]

        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        sync.connect_to(calc, weight_acc)
        sync.connect_to(ref, weight_acc)
        calc.connect_to(first, weight_e, t_min)
        calc.connect_to(calc, -weight_acc)
        ref.connect_to(first, weight_e)
        ref.connect_to(second, weight_e, t_min)
        ref.connect_to(ref, -weight_acc)
        first.connect_to(first, -weight_e)


class MaxPooling(Block):
    def __init__(self, name="pool:", type=Block.output, **kwargs):
        super(MaxPooling, self).__init__(name=name, type=type, **kwargs)
        sync = Neuron(name=name + "sync", parent=self)
        first = Neuron(type=Neuron.output, name=name + "1st", parent=self)
        output = Neuron(type=Neuron.output, name=name + "2nd", parent=self)
        self.neurons = [sync, first, output]
        
        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        sync.connect_to(first, weight_e)

        
class ConvMax(Block):
    def __init__(self, conv_neurons, name="convmax:", type=Block.output, **kwargs):
        super(ConvMax, self).__init__(name=name, type=type, **kwargs)
        first = Neuron(type=Neuron.output, name=name + "1st", parent=self)
        second = Neuron(type=Neuron.output, name=name + "2nd", loihi_type=Neuron.acc, parent=self)
        self.neurons = [first, second]
        self.neurons += conv_neurons

        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        first.connect_to(first, -2*weight_e)
        second.connect_to(second, -weight_acc)
        second.connect_to(first, weight_e)
        for neuron in conv_neurons:
            neuron.connect_to(first, weight_e)
            first.connect_to(neuron, -weight_acc)
            neuron.parent_block = self

        
class Trigger(Block):
    def __init__(self, number, name="pool:", type=Block.trigger, **kwargs):
        super(Trigger, self).__init__(name=name, type=type, **kwargs)
        start_neuron = Neuron(name=self.name + "acc1", loihi_type=Neuron.acc, parent=self)
        delay_neuron = Neuron(name=self.name + "acc2", loihi_type=Neuron.acc, parent=self)
        self.neurons += [start_neuron, delay_neuron]
        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        start_neuron.connect_to(delay_neuron, weight_acc)
        start_neuron.connect_to(start_neuron, -weight_acc)
        delay_neuron.connect_to(delay_neuron, -weight_acc)
        
        for t in range(number):
            trigger_neuron = Neuron(name=self.name + "neuron", type=Neuron.output, parent=self)
            delay_neuron.connect_to(trigger_neuron, weight_e)
            self.neurons += [trigger_neuron]