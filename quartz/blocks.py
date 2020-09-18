from quartz.components import Neuron, Synapse
import quartz
import ipdb


class Block:
    def __init__(self, name='', parent_layer=None):
        self.name = name
        self.neurons = []
        self.parent_layer = parent_layer
    
    def neurons(self):
        return self.neurons

    def _get_neurons_of_type(self, neuron_type):
        return [neuron for neuron in self.neurons if neuron.type == neuron_type]
    
    def input_neurons(self): return self._get_neurons_of_type(Neuron.input)

    def ready_neurons(self): return self._get_neurons_of_type(Neuron.ready)

    def recall_neurons(self): return self._get_neurons_of_type(Neuron.recall)

    def output_neurons(self): return self._get_neurons_of_type(Neuron.output)

    def monitored_neurons(self):
        return [neuron for neuron in self.neurons if neuron.monitor]

    def first(self):
        return self.output_neurons()[0]

    def second(self):
        return self.output_neurons()[1]

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
        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        recall, output = self.neurons
        self.neurons = [] + [recall]
        i = 0
        while(delay>numDendriticAccumulators):
            intermediate = Neuron(name=self.name+"intermediate"+str(i))
            self.neurons[-1].connect_to(intermediate, weight_e, numDendriticAccumulators)
            self.neurons += [intermediate]
            delay -= numDendriticAccumulators
            i += 1
        self.neurons[-1].connect_to(output, weight_e, delay+self.t_min)

        delay = i*self.t_neu
        self.neurons.append(self.neurons.pop(0)) # move recall to the end
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
    def __init__(self, name="split:", promoted=False, monitor=False, **kwargs):
        super(Splitter, self).__init__(name=name, **kwargs)
        input_ = Neuron(type=Neuron.input, promoted=promoted, name=name + "input", monitor=monitor)
        first = Neuron(type=Neuron.output, name=name + "1st", monitor=monitor)
        last = Neuron(type=Neuron.output, name=name + "2nd", monitor=monitor)
        self.neurons = [input_, first, last]
        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        input_.connect_to(first, weight_e)
        first.connect_to(first, -weight_e)
        input_.connect_to(last, 0.5*weight_e)


class ReLCo(Block):
    def __init__(self, inputs, split_input=False, split_output=False, name="relu:", monitor=True, **kwargs):
        super(ReLCo, self).__init__(name=name, **kwargs)
        calc = Neuron(name=name + "calc", monitor=monitor, loihi_type=Neuron.acc)
        sync = Neuron(name=name + "sync", monitor=monitor)
        first = Neuron(name=name + "first", monitor=monitor)
        second = Neuron(name=name + "second", monitor=monitor, loihi_type=Neuron.acc)
        self.neurons = [calc, sync, first, second]
        if split_output:
            first.type = Neuron.output
            second.type = Neuron.output
        else:
            output = Neuron(type=Neuron.output, name=name + "output", monitor=monitor)
            self.neurons += [output]

        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        if split_input:
            for (first_input, second_input, weight) in inputs:
                delay = 5 if weight > 0 else 0
                first_input.connect_to(calc, weight*weight_acc, t_min+delay)
                second_input.connect_to(calc, -weight*weight_acc+delay)
                second_input.connect_to(sync, weight_e/len(inputs)+delay)
        else:
            for (input_, weight) in inputs:
                delay = 5 if weight > 0 else 0
                input_.first().connect_to(calc, weight*weight_acc, t_min+delay)
                input_.second().connect_to(calc, -weight*weight_acc+delay)
                input_.second().connect_to(sync, weight_e/len(inputs)+delay)

        sync.connect_to(calc, weight_acc)
        sync.connect_to(second, weight_acc)
        calc.connect_to(first, weight_e)
        calc.connect_to(calc, -weight_acc)
        second.connect_to(first, weight_e)
        second.connect_to(second, -weight_acc)
        first.connect_to(first, -weight_e)
        if not split_output:
            first.connect_to(output, weight_e)
            second.connect_to(output, weight_e+t_min+t_neu)


class MaxPooling(Block):
    def __init__(self, inputs, split_input=False, extra_delay_first=0, extra_delay_sec=0, split_output=False, name="pool:", monitor=True, **kwargs):
        super(MaxPooling, self).__init__(name=name, **kwargs)
        sync = Neuron(name=name + "sync", monitor=monitor)
        output = Neuron(type=Neuron.output, name=name + "output", monitor=monitor)
        self.neurons = [sync, output]

        weight_e, weight_acc, t_min, t_neu = self.get_params_at_once()
        if split_input:
            for i, (first_input, second_input) in enumerate(inputs):
                acc1 = Neuron(name=name + "acc1_{}".format(i), monitor=monitor, loihi_type=Neuron.acc)
                acc2 = Neuron(name=name + "acc2_{}".format(i), monitor=monitor, loihi_type=Neuron.acc)
                first_input.connect_to(acc1, weight_acc+extra_delay_first)
                second_input.connect_to(acc2, weight_acc+extra_delay_sec)
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
                input_.first().connect_to(acc1, weight_acc+extra_delay_first)
                input_.second().connect_to(acc2, weight_acc+extra_delay_sec)
                acc1.connect_to(acc2, -weight_acc)
                acc1.connect_to(sync, weight_e/len(inputs))
                acc2.connect_to(output, weight_e/len(inputs))
                sync.connect_to(acc2, weight_acc)
                self.neurons += [acc1, acc2]

        if split_output:
            sync.type = Neuron.output
        else:
            sync.connect_to(output, weight_e)
