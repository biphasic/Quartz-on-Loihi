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
        for pre, post, weight, delay in relevant_synapses:
            i = pre_index_list[pre]
            j = post_index_list[post]
            exponent = np.log2(abs(weight)/block.parent_layer.weight_acc)
            if weights[c,j,i] != 0: c+=1
            if exponent > 0 & isinstance(exponent, int):
                weights[c,j,i] = weight / 2**exponent
                exponents[c,j,i] = exponent
            elif exponent <= 1:
                weights[c,j,i] = weight
                exponents[c,j,i] = 0
            else:
                ipdb.set_trace()
            delays[c,j,i] = delay
            
        mask = np.full(weights.shape, False)
        mask[weights!=0] = True
        return weights, delays, exponents, mask


class Input(Block):
    def __init__(self, name="input:", type=Block.output, **kwargs):
        super(Input, self).__init__(name=name, type=type, **kwargs)
        output = Neuron(type=Neuron.output, name=self.name+"neuron", parent=self)
        self.neurons = [output]
        self.output_neurons += [output]
        self.rectifier_neurons += [output]
    
    
class ConstantDelay(Block):
    def __init__(self, value, name="bias:", type=Block.hidden, **kwargs):
        self.value = abs(value)
        super(ConstantDelay, self).__init__(name=name, type=type, **kwargs)
        input_ = Neuron(type=Neuron.input, name=self.name+"input", parent=self)
        output = Neuron(type=Neuron.output, name=self.name+"output", parent=self)
        self.neurons = [input_, output]
        self.input_neurons += [input_]
        self.output_neurons += [output]
        self.layout = False

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
        self.layout = True


class Splitter(Block):
    def __init__(self, name="split:", type=Block.hidden, **kwargs):
        super(Splitter, self).__init__(name=name, type=type, **kwargs)
        input_ = Neuron(type=Neuron.input, name=name + "input", parent=self)
        first = Neuron(type=Neuron.output, name=name + "1st", parent=self)
        last = Neuron(type=Neuron.output, name=name + "2nd", parent=self)
        self.neurons = [input_, first, last]
        self.input_neurons += [input_]
        self.output_neurons += [first, last]
        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        input_.connect_to(first, weight_e)
        first.connect_to(first, -weight_e)
        input_.connect_to(last, 0.5*weight_e)


class ReLCo(Block):
    def __init__(self, name="relco:", type=Block.output, **kwargs):
        super(ReLCo, self).__init__(name=name, type=type, **kwargs)
        calc = Neuron(name=name + "calc", loihi_type=Neuron.acc, type=Neuron.input, parent=self)
        self.neurons = [calc]
        self.input_neurons += [calc]
        self.output_neurons += [calc]
        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        calc.connect_to(calc, -weight_acc) #2**6*

        
class ConvMax(Block):
    def __init__(self, conv_neurons, name="convmax:", type=Block.output, **kwargs):
        super(ConvMax, self).__init__(name=name, type=type, **kwargs)
        first = Neuron(name=name + "1st", type=Neuron.output, parent=self)
        self.neurons = [first]
        self.neurons += conv_neurons
        self.output_neurons += [first]

        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        first.connect_to(first, -8.1*weight_e)
        for neuron in conv_neurons:
            first.connect_to(neuron, -weight_acc) # as an alternative to negative loop from neuron to neuron, this saves some spikes
            neuron.connect_to(first, weight_e)
            neuron.parent_block = self


class Trigger(Block):
    def __init__(self, n_channels, name="pool:", type=Block.trigger, **kwargs):
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


class Output(Block):
    def __init__(self, name="output:", type=Block.output, **kwargs):
        super(Output, self).__init__(name=name, type=type, **kwargs)
        calc = Neuron(name=name + "calc", loihi_type=Neuron.acc, type=Neuron.input, parent=self)
        guard = Neuron(name=name + "guard", parent=self)
        guard2 = Neuron(name=name + "guard2", parent=self)
        sync = Neuron(name=name + "sync", type=Neuron.input, parent=self)
        delay = Neuron(name=name + "delay", loihi_type=Neuron.acc, parent=self)
        first = Neuron(name=name + "1st", type=Neuron.output, parent=self)
        second = Neuron(name=name + "2nd", loihi_type=Neuron.acc, type=Neuron.output, parent=self)
        self.neurons = [calc, sync, guard, guard2, delay, first, second]

        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        sync.connect_to(calc, weight_acc)
        sync.connect_to(delay, weight_acc)
        delay.connect_to(second, weight_acc)
        delay.connect_to(delay, -weight_acc)
        calc.connect_to(guard, weight_e)
        calc.connect_to(calc, -weight_acc)
        guard.connect_to(guard2, weight_e)
        guard.connect_to(guard, -3.1*weight_e)
        guard2.connect_to(first, weight_e)
        guard2.connect_to(guard2, -3.1*weight_e)
        first.connect_to(first, -3.1*weight_e)
        second.connect_to(second, -weight_acc)
