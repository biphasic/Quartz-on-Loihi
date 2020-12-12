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

    def rectifier_neurons(self): return self._get_neurons_of_type(Neuron.rectifier)

    def neuron(self, name):
        res = tuple(neuron for neuron in self.neurons if name in neuron.name)
        if len(res) > 1:
            [print(neuron.name) for neuron in res]
            raise Exception("Provided name was ambiguous, results returned: ")
        elif len(res) == 0:
            raise Exception("No neuron with name {} found. Available names are: {}".format(name, self.neurons))
        return res[0]
    
    def first(self): 
        neuron = self.output_neurons()[0]
        assert "1st" in neuron.name or isinstance(self, quartz.blocks.Input)
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

    def n_outgoing_connections(self):
        n_conns = 0
        for neuron in self.neurons:
            for synapse in neuron.synapses['pre']:
                if synapse.pre.parent_block != synapse.post.parent_block: n_conns += 1
        return n_conns
        #return sum([len(neuron.synapses['post']) for neuron in self.neurons])
    
    def n_incoming_connections(self):
        return sum([len(neuron.synapses['post']) for neuron in self.neurons])
    
    def n_recurrent_connections(self):
        n_conns = 0
        for neuron in self.neurons:
            for synapse in neuron.synapses['pre']:
                if synapse.pre.parent_block == synapse.post.parent_block: n_conns += 1
        return n_conns

    def get_params_at_once(self):
        return self.parent_layer.get_params_at_once()
    
    def get_connected_blocks(self):
        connected_blocks = []
        for neuron in self.neurons:
            for synapse in neuron.incoming_synapses():
                connected_blocks.append(synapse.pre.parent_block)
            for synapse in neuron.outgoing_synapses():
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
        weights = np.zeros((1, len(block.neurons), len(self.neurons)))
        delays = np.zeros_like(weights)
        for i, neuron in enumerate(self.neurons):
            for synapse in neuron.outgoing_synapses():
                if synapse.post in block.neurons:
                    j = block.neurons.index(synapse.post)
                    for m in range(weights.shape[0]): 
                        # multiple weight matrices are possible for multiple connection from one to the same neuron
                        if weights[m, j, i] != 0 and (m+1) == weights.shape[0]:
                            weights = np.resize(weights, (weights.shape[0]+1, *weights.shape[1:]))
                            weights[-1] = 0
                            delays = np.resize(delays, (delays.shape[0]+1, *delays.shape[1:]))
                            delays[-1] = 0
                            weights[-1, j, i] = synapse.weight
                            delays[-1, j, i] = synapse.delay
                            break
                        if weights[m, j, i] != 0:
                            continue
                        else:
                            weights[m, j, i] = synapse.weight
                            delays[m, j, i] = synapse.delay
                            break
        mask = np.zeros_like(weights)
        mask[weights!=0] = 1
        return weights, delays, mask


class Input(Block):
    def __init__(self, name="input:", type=Block.output, **kwargs):
        super(Input, self).__init__(name=name, type=type, **kwargs)
        output = Neuron(type=Neuron.output, name=self.name+"neuron", parent=self)
        self.neurons = [output]
    
    
class ConstantDelay(Block):
    def __init__(self, value, name="bias:", type=Block.hidden, **kwargs):
        self.value = abs(value)
        super(ConstantDelay, self).__init__(name=name, type=type, **kwargs)
        input_ = Neuron(type=Neuron.input, name=self.name+"input", parent=self)
        output = Neuron(type=Neuron.output, name=self.name+"output", parent=self)
        self.neurons = [input_, output]
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
        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        input_.connect_to(first, weight_e)
        first.connect_to(first, -weight_e)
        input_.connect_to(last, 0.5*weight_e)


class ReLCo(Block):
    def __init__(self, name="relco:", type=Block.output, **kwargs):
        super(ReLCo, self).__init__(name=name, type=type, **kwargs)
        calc = Neuron(name=name + "calc", loihi_type=Neuron.acc, type=Neuron.input, parent=self)
        first = Neuron(name=name + "1st", type=Neuron.output, parent=self)
        self.neurons = [calc, first]

        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        calc.connect_to(calc, -weight_acc)
        calc.connect_to(first, weight_e)
        first.connect_to(first, -weight_e)


class WTA(Block):
    def __init__(self, name="pool:", type=Block.output, **kwargs):
        super(WTA, self).__init__(name=name, type=type, **kwargs)
        sync = Neuron(name=name + "sync", parent=self)
        first = Neuron(type=Neuron.output, name=name + "1st", parent=self)
        output = Neuron(type=Neuron.output, name=name + "2nd", parent=self)
        self.neurons = [sync, first, output]
        
        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        sync.connect_to(first, weight_e)

        
class ConvMax(Block):
    def __init__(self, conv_neurons, name="convmax:", type=Block.output, **kwargs):
        super(ConvMax, self).__init__(name=name, type=type, **kwargs)
        first = Neuron(name=name + "1st", type=Neuron.output, parent=self)
        self.neurons = [first]
        self.neurons += conv_neurons

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
        assert n_channels > 0
        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        rect.connect_to(rect, -weight_acc)
        for t in range(n_channels):
            trigger_neuron = Neuron(name=self.name + "trigger" + str(t), type=Neuron.output, loihi_type=Neuron.acc, parent=self)
            trigger_neuron.connect_to(trigger_neuron, -weight_acc)
            self.neurons += [trigger_neuron]
        self.neurons[-1].connect_to(rect, weight_acc)


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
