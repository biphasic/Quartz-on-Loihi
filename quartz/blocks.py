import numpy as np
from quartz.components import Neuron, Synapse
import quartz
import ipdb


class Connection:
    def __init__(self, pre, post, weight, delay):
        self.name = "{0} --({1};{2})-->\t{3}".format(
            pre.name, weight, delay, post.name
        )
        if delay is None: delay = np.zeros_like(weight)
        self.pre = pre
        self.post = post
        self.weight = weight
        self.delay = delay

    def __repr__(self):
        return self.name
    
    
class Block:
    input, hidden, output = range(3)
    
    def __init__(self, name, type, parent_layer):
        self.name = name
        self.type = type
        self.neurons = []
        self.parent_layer = parent_layer
        self.connections = {"pre": [], "post": []}
    
    def neurons(self):
        return self.neurons

    def _get_neurons_of_type(self, neuron_type):
        return [neuron for neuron in self.neurons if neuron.type == neuron_type]
    
    def input_neurons(self): return self._get_neurons_of_type(Neuron.input)

    def output_neurons(self): return self._get_neurons_of_type(Neuron.output)

    def monitored_neurons(self):
        return [neuron for neuron in self.neurons if neuron.monitor]

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
    
    def connect_to(self, target_block, weight, delay=None):
        self.connections["pre"].append(
            Connection(pre=self, post=target_block, weight=weight, delay=delay)
        )
        target_block.connect_from(self, weight, delay)

    def connect_from(self, source_neuron, weight, delay):
        self.connections["post"].append(
            Connection(pre=source_neuron, post=self, weight=weight, delay=delay)
        )

    def incoming_connections(self):
        return self.connections["post"]

    def outgoing_connections(self):
        return self.connections["pre"]

    def has_incoming_connections(self):
        return self.connections["post"] != []

#     def internal_weight_matrix(self): ipdb.set_trace()
#     def internal_delay_matrix(self): pass
#     def internal_connection_matrix(self):
#         ipdb.set_trace()


class ConstantDelay(Block):
    def __init__(self, value, name="bias:", type=Block.hidden, monitor=False, **kwargs):
        self.value = abs(value)
        self.name = name
        self.monitor = monitor
        super(ConstantDelay, self).__init__(name=name, type=type, **kwargs)
        input_ = Neuron(type=Neuron.input, name=self.name+"input", monitor=self.monitor)
        output = Neuron(type=Neuron.output, name=self.name+"output", monitor=self.monitor)
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

    def layout_delays(self, t_max, numDendriticAccumulators):
        delay = round(self.value * t_max)
        numDendriticAccumulators = numDendriticAccumulators-2
        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        input_, output = self.neurons
        self.neurons = [] + [input_]
        i = 0
        while(delay>numDendriticAccumulators):
            intermediate = Neuron(name=self.name+"intermediate"+str(i))
            self.neurons[-1].connect_to(intermediate, weight_e, numDendriticAccumulators)
            self.neurons += [intermediate]
            delay -= numDendriticAccumulators
            i += 1
        self.neurons[-1].connect_to(output, weight_e, delay+t_min)

        delay = i*t_neu
        self.neurons.append(self.neurons.pop(0)) # move input_ to the end
        i = 0
        while(delay>numDendriticAccumulators):
            intermediate = Neuron(name=self.name+"intermediate-output"+str(i))
            self.neurons[-1].connect_to(intermediate, weight_e, numDendriticAccumulators)
            self.neurons += [intermediate]
            delay -= (numDendriticAccumulators+1)
            i += 1
        self.neurons[-1].connect_to(output, weight_e, delay)
        self.neurons += [output]


class Splitter(Block):
    def __init__(self, name="split:", type=Block.hidden, monitor=False, **kwargs):
        super(Splitter, self).__init__(name=name, type=type, **kwargs)
        input_ = Neuron(type=Neuron.input, name=name + "input", monitor=monitor)
        first = Neuron(type=Neuron.output, name=name + "1st", monitor=monitor)
        last = Neuron(type=Neuron.output, name=name + "2nd", monitor=monitor)
        self.neurons = [input_, first, last]
        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        input_.connect_to(first, weight_e)
        first.connect_to(first, -weight_e)
        input_.connect_to(last, 0.5*weight_e)


class ReLCo(Block):
    def __init__(self, name="relco:", type=Block.output, monitor=False, **kwargs):
        super(ReLCo, self).__init__(name=name, type=type, **kwargs)
        calc = Neuron(name=name + "calc", monitor=monitor, loihi_type=Neuron.acc, type=Neuron.input)
        sync = Neuron(name=name + "sync", monitor=monitor, type=Neuron.input)
        first = Neuron(name=name + "first", monitor=monitor, type=Neuron.output)
        second = Neuron(name=name + "second", monitor=monitor, loihi_type=Neuron.acc, type=Neuron.output)
        self.neurons = [calc, sync, first, second]

        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        sync.connect_to(calc, weight_acc)
        sync.connect_to(second, weight_acc)
        calc.connect_to(first, weight_e)
        calc.connect_to(calc, -weight_acc)
        second.connect_to(first, weight_e)
        second.connect_to(second, -weight_acc)
        first.connect_to(first, -weight_e)


class MaxPooling(Block):
    def __init__(self, extra_delay_first=0, extra_delay_sec=0, split_output=False, name="pool:", type=Block.output, monitor=False, **kwargs):
        super(MaxPooling, self).__init__(name=name, **kwargs)
        sync = Neuron(name=name + "sync", monitor=monitor)
        output = Neuron(type=Neuron.output, name=name + "output", monitor=monitor)
        self.neurons = [sync, output]

        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        if split_input:
            for i, (first_input, second_input) in enumerate(inputs):
                acc1 = Neuron(name=name + "acc1_{}".format(i), monitor=monitor, loihi_type=Neuron.acc)
                acc2 = Neuron(name=name + "acc2_{}".format(i), monitor=monitor, loihi_type=Neuron.acc)
                first_input.connect_to(acc1, weight_acc, extra_delay_first)
                second_input.connect_to(acc2, weight_acc, extra_delay_sec)
                acc1.connect_to(acc2, -weight_acc)
                acc1.connect_to(sync, weight_e/len(inputs))
                acc1.connect_to(acc1, -weight_acc)
                acc2.connect_to(output, weight_e/len(inputs))
                acc2.connect_to(acc2, -weight_acc)
                sync.connect_to(acc2, weight_acc)
                self.neurons += [acc1, acc2]
        else:
            for i, input_ in enumerate(inputs):
                acc1 = Neuron(name=name + "acc1_{}".format(i), monitor=monitor)
                acc2 = Neuron(name=name + "acc2_{}".format(i), monitor=monitor)
                input_.first().connect_to(acc1, weight_acc, extra_delay_first)
                input_.second().connect_to(acc2, weight_acc, extra_delay_sec)
                acc1.connect_to(acc2, -weight_acc)
                acc1.connect_to(sync, weight_e/len(inputs))
                acc2.connect_to(output, weight_e/len(inputs))
                sync.connect_to(acc2, weight_acc)
                self.neurons += [acc1, acc2]

        if split_output:
            sync.type = Neuron.output
        else:
            sync.connect_to(output, weight_e)
