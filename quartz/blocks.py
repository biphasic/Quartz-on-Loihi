from quartz.components import Neuron, Synapse
import quartz
import ipdb


class Block:
    def __init__(self, name='', weight_e=120, weight_acc=2**6, t_syn=0, t_min=1, t_neu=1):
        self.name = name
        self.neurons = []
        self.blocks = []
        self.weight_e = weight_e
        self.weight_acc = weight_acc
        self.weight_exp = weight_acc
        self.weight_gate = weight_e
        self.t_syn = t_syn
        self.t_min = t_min
        self.t_neu = t_neu

    def neuron(self, name):
        res = tuple(neuron for neuron in self.neurons if name in neuron.name)
        if len(res) > 1:
            [print(neuron.name) for neuron in res]
            raise Exception("Provided name was ambiguous, results returned: ")
        elif len(res) == 0:
            raise Exception("No neuron with name {} found. Available names are: {}".format(name, self.neurons))
        return res[0]

    # notice the difference: promoted neurons in all_neurons, otherwise self.neurons
    def _get_neurons_of_type(self, neuron_type, promoted):
        if promoted: return [neuron for neuron in self.all_neurons() if neuron.promoted and neuron.type == neuron_type]
        return [neuron for neuron in self.neurons if neuron.type == neuron_type]
    
    def input_neurons(self, promoted=False): return self._get_neurons_of_type(Neuron.input, promoted)

    def ready_neurons(self, promoted=False): return self._get_neurons_of_type(Neuron.ready, promoted)

    def recall_neurons(self, promoted=False): return self._get_neurons_of_type(Neuron.recall, promoted)

    def output_neurons(self, promoted=False): return self._get_neurons_of_type(Neuron.output, promoted)

    def monitored_neurons(self):
        return tuple(neuron for neuron in self.all_neurons() if neuron.monitor)

    def first(self):
        return self.output_neurons()[0]

    def second(self):
        return self.output_neurons()[1]

    def neurons_from_block(self, network, neurons):
        if network.blocks == []: # base case
            return neurons + network.neurons
        branch_neurons = [] + network.neurons
        for single_block in network.blocks:
            branch_neurons += self.neurons_from_block(single_block, neurons)
        return neurons + branch_neurons

    def check_all_blocks_for_delays(self, network, t_max, numDendriticAccumulators):
        if isinstance(network, stick.ConstantDelay):
            network.reset()
            network.layout_delays(t_max, numDendriticAccumulators)
        for single_block in network.blocks:
            self.check_all_blocks_for_delays(single_block, t_max, numDendriticAccumulators)

    def connect_signed_output_to_input(self, block, weight, delay, target_input=0):
        self.output_neurons()[0].connect_to(block.input_neurons()[target_input*2], weight, delay)
        self.output_neurons()[1].connect_to(block.input_neurons()[target_input*2+1], weight, delay)

    def all_neurons(self):
        return self.neurons_from_block(self, [])

    def print_connections(self, maximum=10e7):
        for i, neuron in enumerate(self.all_neurons()):
            if neuron.synapses['pre'] != []: 
                [print(connection) for connection in neuron.synapses['pre']]
            elif neuron.type != Neuron.output and neuron.type != Neuron.ready:
                print("Warning: neuron {} seems not to be connected to any other neuron.".format(neuron.name))
            if i > maximum: break

    def _count_connections(self, neurons):
        n_connections = 0
        for neuron in neurons:
            n_connections += len(neuron.synapses['post'])
        return n_connections

    def n_connections(self):
        return self._count_connections(self.neurons)

    def n_all_connections(self):
        return self._count_connections(self.all_neurons())

    def get_params_at_once(self):
        return self.weight_e, self.weight_acc, self.t_syn, self.t_min, self.t_neu


class ConstantDelay(Block):
    def __init__(self, value, name="constDelay:", promoted=False, monitor=False, **kwargs):
        self.value = abs(value)
        self.name = name
        self.promoted = promoted
        self.monitor = monitor
        super(ConstantDelay, self).__init__(name=name, **kwargs)
        recall = Neuron(type=Neuron.recall, promoted=self.promoted, name=self.name+"recall", monitor=self.monitor)
        output = Neuron(type=Neuron.output, name=self.name+"output", monitor=self.monitor)
        self.neurons = [recall, output]            
        self.reset()

    def reset(self):
        if len(self.neurons) > 2:
            recall = self.neurons[0]
            recall.reset_outgoing_connections()
            output = self.neurons[-1]
            output.reset_incoming_connections()
            self.neurons = [recall, output]            
        else:
            pass

    def layout_delays(self, t_max, numDendriticAccumulators):
        delay = round(self.value * t_max)
        numDendriticAccumulators = numDendriticAccumulators-2
        recall, output = self.neurons
        self.neurons = [] + [recall]
        i = 0
        while(delay>numDendriticAccumulators):
            intermediate = Neuron(name=self.name+"intermediate"+str(i))
            self.neurons[-1].connect_to(intermediate, self.weight_e, self.t_syn + numDendriticAccumulators)
            self.neurons += [intermediate]
            delay -= numDendriticAccumulators
            i += 1
        self.neurons[-1].connect_to(output, self.weight_e, delay+self.t_min)

        delay = i*self.t_neu
        self.neurons.append(self.neurons.pop(0)) # move recall to the end
        i = 0
        while(delay>numDendriticAccumulators):
            intermediate = Neuron(name=self.name+"intermediate-output"+str(i))
            self.neurons[-1].connect_to(intermediate, self.weight_e, self.t_syn + numDendriticAccumulators)
            self.neurons += [intermediate]
            delay -= (numDendriticAccumulators+1)
            i += 1
        self.neurons[-1].connect_to(output, self.weight_e, delay)
        self.neurons += [output]
#         if self.monitor:
#             print("New connections for " + self.name)
#             self.print_connections()


class Splitter(Block):
    def __init__(self, name="split:", promoted=False, monitor=False, **kwargs):
        super(Splitter, self).__init__(name=name, **kwargs)
        input_ = Neuron(type=Neuron.input, promoted=promoted, name=name + "input", monitor=monitor)
        first = Neuron(type=Neuron.output, name=name + "1st", monitor=monitor)
        last = Neuron(type=Neuron.output, name=name + "2nd", monitor=monitor)
        self.neurons = [input_, first, last]
        weight_e, weight_acc, t_syn, t_min, t_neu = self.get_params_at_once()
        input_.connect_to(first, weight_e, t_syn)
        first.connect_to(first, -weight_e, t_syn)
        input_.connect_to(last, 0.5*weight_e, t_syn)


class ReLCo(Block):
    def __init__(self, inputs, split_input=False, split_output=False, name="relu:", monitor=True, **kwargs):
        super(ReLCo, self).__init__(name=name, **kwargs)
        calc = Neuron(name=name + "calc", monitor=monitor)
        sync = Neuron(name=name + "sync", monitor=monitor)
        first = Neuron(name=name + "first", monitor=monitor)
        second = Neuron(name=name + "second", monitor=monitor)
        self.neurons = [calc, sync, first, second]
        if split_output:
            first.type = Neuron.output
            second.type = Neuron.output
        else:
            output = Neuron(type=Neuron.output, name=name + "output", monitor=monitor)
            self.neurons += [output]

        weight_e, weight_acc, t_syn, t_min, t_neu = self.get_params_at_once()
        if split_input:
            for (first_input, second_input, weight) in inputs:
                delay = 5 if weight > 0 else 0
                first_input.connect_to(calc, weight*weight_acc, t_min+t_syn+delay, type=Synapse.ge)
                second_input.connect_to(calc, -weight*weight_acc, t_syn+delay, type=Synapse.ge)
                second_input.connect_to(sync, weight_e/len(inputs), t_syn+delay)
        else:
            for (input_, weight) in inputs:
                delay = 5 if weight > 0 else 0
                input_.first().connect_to(calc, weight*weight_acc, t_min+t_syn+delay, type=Synapse.ge)
                input_.second().connect_to(calc, -weight*weight_acc, t_syn+delay, type=Synapse.ge)
                input_.second().connect_to(sync, weight_e/len(inputs), t_syn+delay)

        sync.connect_to(calc, weight_acc, t_syn, type=Synapse.ge)
        sync.connect_to(second, weight_acc, t_syn, type=Synapse.ge)
        calc.connect_to(first, weight_e, t_syn)
        second.connect_to(first, weight_e, t_syn)
        first.connect_to(first, -weight_e, t_syn)
        if not split_output:
            first.connect_to(output, weight_e, t_syn)
            second.connect_to(output, weight_e, t_syn+t_min+t_neu)


class MaxPooling(Block):
    def __init__(self, inputs, split_input=False, extra_delay_first=0, extra_delay_sec=0, split_output=False, name="pool:", monitor=True, **kwargs):
        super(MaxPooling, self).__init__(name=name, **kwargs)
        sync = Neuron(name=name + "sync", monitor=monitor)
        output = Neuron(type=Neuron.output, name=name + "output", monitor=monitor)
        self.neurons = [sync, output]

        weight_e, weight_acc, t_syn, t_min, t_neu = self.get_params_at_once()
        if split_input:
            for i, (first_input, second_input) in enumerate(inputs):
                acc1 = Neuron(name=name + "acc1_{}".format(i), monitor=monitor)
                acc2 = Neuron(name=name + "acc2_{}".format(i), monitor=monitor)
                first_input.connect_to(acc1, weight_acc, t_syn+extra_delay_first, type=Synapse.ge)
                second_input.connect_to(acc2, weight_acc, t_syn+extra_delay_sec, type=Synapse.ge)
                acc1.connect_to(acc2, -weight_acc, t_syn, type=Synapse.ge)
                acc1.connect_to(sync, weight_e/len(inputs), t_syn)
                acc2.connect_to(output, weight_e/len(inputs), t_syn)
                sync.connect_to(acc2, weight_acc, t_syn, type=Synapse.ge)
                self.neurons += [acc1, acc2]
        else:
            for i, input_ in enumerate(inputs):
                acc1 = Neuron(name=name + "acc1_{}".format(i), monitor=monitor)
                acc2 = Neuron(name=name + "acc2_{}".format(i), monitor=monitor)
                input_.first().connect_to(acc1, weight_acc, t_syn+extra_delay_first, type=Synapse.ge)
                input_.second().connect_to(acc2, weight_acc, t_syn+extra_delay_sec, type=Synapse.ge)
                acc1.connect_to(acc2, -weight_acc, t_syn, type=Synapse.ge)
                acc1.connect_to(sync, weight_e/len(inputs), t_syn)
                acc2.connect_to(output, weight_e/len(inputs), t_syn)
                sync.connect_to(acc2, weight_acc, t_syn, type=Synapse.ge)
                self.neurons += [acc1, acc2]

        if split_output:
            sync.type = Neuron.output
        else:
            sync.connect_to(output, weight_e, t_syn)
