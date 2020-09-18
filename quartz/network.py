from quartz.components import Neuron, Synapse
from quartz.utils import decode_spike_timings
import quartz
import numpy as np
import math
import nxsdk
import nxsdk.api.n2a as nx
from nxsdk.graph.monitor.probes import PerformanceProbeCondition, IntervalProbeCondition, SpikeProbeCondition
from nxsdk.logutils.nxlogging import set_verbosity, LoggingLevel
import datetime
import ipdb


class Network:
    def __init__(self, layers, name=''):
        self.name = name
        self.data = {}
        self.layers = layers
        for i in range(1, len(layers)):
            layers[i].connect_from(layers[i-1])
        
    def __call__(self, input_spike_list, t_max, vth_mant=2**16, num_chips=1):
        board, probes = self.build_model(input_spike_list, t_max)
        self.run_on_loihi(board, probes)

    def build_model(input_spike_list, t_max):
        net = nx.NxNet()
        assert np.log2(t_max).is_integer()
        # add intermediate neurons for delay encoder depending on t_max
        self.check_block_delays(t_max, 2**3)
        # assign core layout based on no of compartments and no of unique connections
        self.check_layout()
        # create loihi compartments
        
        # connect loihi compartments
        weight_exponent = np.log2(vth_mant/(t_max*self.model.weight_acc))
        
        # add inputs
            
    def n_compartments(self):
        return sum([layer.n_compartments() for layer in self.layers])
    
    def n_parameters(self):
        return sum([layer.n_parameters() for layer in self.layers])
    
    def n_connections(self):
        return sum([layer.n_connections() for layer in self.layers])
    
    def check_block_delays(self, t_max, numDendriticAccumulators):
        for layer in self.layers:
            layer.check_block_delays(t_max, numDendriticAccumulators)

    def check_layout(self):
        n_compartments = [layer.n_compartments() for layer in self.layers]
        n_connections = [layer.n_connections() for layer in self.layers]
        n_parameters = [layer.n_parameters() for layer in self.layers]

    def __repr__(self):
        return '\n'.join([layer.name for layer in self.layers])
    
    
    #@profile
    def run_on_loihi(self, run_time, input_spike_list=None, recall_spike_list=None, t_max=None, vth_mant=2**16, full_probes=True, num_chips=1,
                     partition='loihi', probe_selection=[], probe_interval=1, profiling=False, plot=True, logging=False, save_plot=False):
        board, probes = self.build_loihi_model(input_spike_list, recall_spike_list, t_max, vth_mant, logging,
                                               full_probes, num_chips, probe_selection, probe_interval, profiling)
        return self.run_model(board, probes, run_time, t_max, profiling, partition, plot, save_plot)
        
    def build_loihi_model(self, input_spike_list=None, recall_spike_list=None, t_max=None, vth_mant=2**16, logging=False, full_probes=True, 
                          num_chips=1, probe_selection=[], probe_interval=1, profiling=False):
        net = nx.NxNet()
        assert np.log2(t_max).is_integer()

        # regulate precision of accumulating neurons depending on t_max, by modifying weight exponent
        weight_exponent = np.log2(vth_mant/(t_max*self.model.weight_acc))
        print("weight exponent: {}".format(weight_exponent))
        weight_exponent_multicomp = weight_exponent+1
        #print("weight_exponent_multicomp: {}".format(weight_exponent_multicomp))

        numDendriticAccumulators = 2**3 # maximum number to support automatic compilation
        # add intermediate neurons for delay encoder depending on t_max just before execution on Loihi
        self.model.check_all_blocks_for_delays(self.model, t_max, numDendriticAccumulators)

        quartz_neurons = list(set(self.model.all_neurons())) # flattened network

        print("Before delay check: {} neurons.".format(len(quartz_neurons)))
        # build loihi model
        for neuron in quartz_neurons:
            synapse_list = neuron.outgoing_synapses().copy()
            for synapse in synapse_list:
                if synapse.delay > 6:
                    delay = synapse.delay
                    accumulators = numDendriticAccumulators-2
                    origin = synapse.pre
                    target = synapse.post
                    origin.remove_connections_to(target)
                    intermediates = [] + [origin]
                    i = 0
                    while(delay>accumulators):
                        intermediate = Neuron(name=neuron.name+"-intermediate"+str(i), loihi_type=Neuron.pulse)
                        intermediates[-1].connect_to(intermediate, self.model.weight_e, self.model.t_syn + accumulators)
                        intermediates += [intermediate]
                        delay -= (accumulators + 1)
                        i += 1
                    intermediates[-1].connect_to(target, synapse.weight, delay, type=synapse.type)
                    intermediates.remove(origin)
                    quartz_neurons += intermediates

            if neuron.has_incoming_synapses() and all(synapse.type == Synapse.ge for synapse in neuron.incoming_synapses()):
                neuron.loihi_type = Neuron.acc
            elif neuron.has_incoming_synapses() and any(synapse.type == Synapse.gf for synapse in neuron.incoming_synapses()):
                neuron.loihi_type = Neuron.multi
            else:
                neuron.loihi_type = Neuron.pulse
        print("After delay check: {} neurons.".format(len(quartz_neurons)))
    
        # create loihi neurons. If a neuron has only incoming ge synapses, then the compartment current type will be constant. 
        # The default however is an instantaneous current decay, matching a V synapse.
        net_neurons = []
        n_cores = num_chips * 128
        n_loihi_compartments = np.zeros((n_cores))
        n_synapses = np.zeros((n_cores))
        for neuron in quartz_neurons:
            loihi_core_index = int(np.random.rand() * n_cores)
            tries = 0
            while n_synapses[loihi_core_index] + len(neuron.incoming_synapses()) > 4096:
                loihi_core_index = int(np.random.rand() * n_cores)
                tries += 1
                if tries > 3 * n_cores: raise Exception("Too many synapses everywhere on all cores. Try increasing the number of chips used.")
            n_synapses[loihi_core_index] += len(neuron.incoming_synapses())
            if neuron.loihi_type == Neuron.acc:
                loihi_neuron = net.createCompartment(nx.CompartmentPrototype(logicalCoreId=loihi_core_index, vThMant=vth_mant, compartmentCurrentDecay=0))
                # inhibitory connection to reset current to 0 after spike
                loihi_neuron.connect(loihi_neuron, prototype=nx.ConnectionPrototype(weight=-self.model.weight_acc, weightExponent=weight_exponent, signMode=3))
            elif neuron.loihi_type == Neuron.multi:
                loihi_neuron = net.createNeuron(nx.NeuronPrototype(self.model.get_neuron_prototype_on_core(loihi_core_index, vth_mant, t_max, pulse_mant)))
                n_loihi_compartments[loihi_core_index] += 5
                # self-excitatory connection to keep gate open
                loihi_neuron.dendrites[0].dendrites[1].connect(loihi_neuron.dendrites[0].dendrites[1], 
                                                               prototype=nx.ConnectionPrototype(weight=self.model.weight_gate, 
                                                                                                weightExponent=weight_exponent_multicomp))
            else: # standard pulse neuron. most common
                no_inputs = len(neuron.incoming_synapses())
                pulse_mant = (self.model.weight_e - 1) * 2**weight_exponent
                if no_inputs > 0:
                    assert no_inputs <= self.model.weight_e
                    ratio = (self.model.weight_e // no_inputs) * no_inputs / self.model.weight_e # hack for sync neurons
                    # if no_inputs > 300: ipdb.set_trace()
                    if ratio != 1:
                        pulse_mant = self.model.weight_e * ratio * 2**weight_exponent - 1
                loihi_neuron = net.createCompartment(nx.CompartmentPrototype(logicalCoreId=loihi_core_index, vThMant=pulse_mant, 
                                                                             compartmentCurrentDecay=2**12-1))
            n_loihi_compartments[loihi_core_index] += 1
            net_neurons.append(loihi_neuron)

        print("{} Loihi neuron creation done, now connecting...".format(datetime.datetime.now()))
        # connect neurons like in the higher level model. Weights and delays are directly transferable. 
        for i, neuron in enumerate(net_neurons):
            for synapse in quartz_neurons[i].synapses['pre']:
                index_target_neuron = quartz_neurons.index(synapse.post)
                target_neuron = net_neurons[index_target_neuron]
                # choose different target dendrites for multicompartment neurons
                anti_target = None
                if synapse.post.loihi_type == Neuron.multi: # isinstance(target_neuron, nxsdk.net.nodes.neurons.Neuron):
                    if synapse.type == Synapse.gf:
                        anti_target = target_neuron.dendrites[0].dendrites[0].dendrites[1]
                        target_neuron = target_neuron.dendrites[0].dendrites[0].dendrites[0]
                        extra_exponent = 7 - weight_exponent_multicomp 
                        exponent = weight_exponent_multicomp+extra_exponent
                    elif synapse.type == Synapse.ge:
                        target_neuron = target_neuron.dendrites[1]
                        exponent = weight_exponent_multicomp
                    elif synapse.type == Synapse.gate:
                        target_neuron = target_neuron.dendrites[0].dendrites[1]
                        exponent = weight_exponent_multicomp
                    else: raise NotImplementedError("Synapse type not supported")
                elif synapse.post.loihi_type == Neuron.pulse and (synapse.weight < -256 or synapse.weight > 255):
                    synapse.weight = math.ceil(synapse.weight / 2)
                    exponent = weight_exponent + 1
                else:
                    exponent = weight_exponent
                    
                sign_mode = 2 if synapse.weight >= 0 else 3
                if (synapse.weight < -256 or synapse.weight > 255): ipdb.set_trace()
                neuron.connect(target_neuron, prototype=nx.ConnectionPrototype(weight=round(synapse.weight), delay=synapse.delay,
                                                                               weightExponent=exponent, signMode=sign_mode,))
                # The following is a hack for nonlinear neurons. since exponential decay over many time steps is very inaccurate for small voltages, 
                # I have an artificially big voltage and subtract a minor big artificial voltage that decays as well. 
                # That way I get a fine-grained exponential decay for a low voltage over many time steps. 
                if anti_target != None: 
                    the_weight = -(1 - 2**(-extra_exponent))*synapse.weight
                    assert the_weight.is_integer()
                    neuron.connect(anti_target, prototype=nx.ConnectionPrototype(weight=the_weight, delay=synapse.delay,
                                                                               weightExponent=exponent, signMode=3,))

        # spike generators that take input spikes
        if input_spike_list != None: net = self.add_spike_generators(input_spike_list, self.model.input_neurons, net, quartz_neurons, net_neurons, weight_exponent)
        # spike generators that take recall spikes
        if recall_spike_list!= None: net = self.add_spike_generators(recall_spike_list, self.model.recall_neurons, net, quartz_neurons, net_neurons, weight_exponent)


        # put probes with conditions on neurons that have the monitor parameter enabled and which are in the probe_selection list
        if not profiling:
            vol_probe_cond = nx.IntervalProbeCondition(dt=probe_interval)
            spike_probe_cond = nx.SpikeProbeCondition(dt=probe_interval)
            measurements = [nx.ProbeParameter.SPIKE]
            conditions = [spike_probe_cond]
            if full_probes:
                measurements += [nx.ProbeParameter.COMPARTMENT_VOLTAGE, nx.ProbeParameter.COMPARTMENT_CURRENT]
                conditions += [vol_probe_cond, vol_probe_cond]
            probe_selection += ["output"]

            name_probes_dicts = {quartz_neurons[i].name:neuron.probe(measurements, probeConditions=conditions)\
                                 for i, neuron in enumerate(net_neurons)\
                                 if quartz_neurons[i].monitor and not isinstance(net_neurons[i], nxsdk.net.nodes.neurons.Neuron) 
                                 and any([string in quartz_neurons[i].name for string in probe_selection])}

            # probes for multicompartment neurons
            for i, neuron in enumerate(net_neurons):
                if quartz_neurons[i].monitor and isinstance(net_neurons[i], nxsdk.net.nodes.neurons.Neuron)\
                and any([string in quartz_neurons[i].name for string in probe_selection]):
                    name_probes_dicts[quartz_neurons[i].name+"_soma"] = neuron.soma.probe(measurements, probeConditions=conditions)
                    name_probes_dicts[quartz_neurons[i].name+"_const"] = neuron.dendrites[1].probe(measurements, probeConditions=conditions)
                    name_probes_dicts[quartz_neurons[i].name+"_exp_sum"] = neuron.dendrites[0].dendrites[0].probe(measurements, probeConditions=conditions)
                    name_probes_dicts[quartz_neurons[i].name+"_exp_pos"] = \
                                neuron.dendrites[0].dendrites[0].dendrites[0].probe(measurements, probeConditions=conditions)
                    name_probes_dicts[quartz_neurons[i].name+"_exp_neg"] = \
                                neuron.dendrites[0].dendrites[0].dendrites[1].probe(measurements, probeConditions=conditions)
                    #name_probes_dicts[quartz_neurons[i].name+"_gate"] = neuron.dendrites[0].dendrites[1].probe(measurements, probeConditions=conditions)
        
        if not logging:
            set_verbosity(LoggingLevel.ERROR)
        else:
            print("This network uses {} neurons and {} probes on {} Loihi compartments. Params: eta={} and Vth={}"\
                  .format(len(quartz_neurons), len(name_probes_dicts), n_loihi_compartments, weight_exponent, vth_mant*2**6))

        print("{} Compiling now...".format(datetime.datetime.now()))
        #ipdb.set_trace()
        board = nx.N2Compiler().compile(net)
        #ipdb.set_trace()
        return board, name_probes_dicts


    def run_model(self, board, probes, run_time, t_max, profiling=False, partition='loihi', plot=True, save_plot=False):
        name_probes_dicts = probes

        if profiling:
            profile_probe_cond = PerformanceProbeCondition(tStart=1, tEnd=run_time, bufferSize=1000, binSize=100)
            time_probe = board.probe(nx.ProbeParameter.EXECUTION_TIME, profile_probe_cond)
            energy_probe = board.probe(nx.ProbeParameter.ENERGY, profile_probe_cond)
        board.run(run_time, partition=partition)
        board.disconnect()
        
        # print(board.energyTimeMonitor.powerProfileStats.power)

        if profiling: 
            total_spike_time = int(time_probe.totalSpikingTime)
            total_spike_energy = energy_probe.totalSpikingPhaseEnergy
            print("total time spent in sec on execution: {}s, host: {}s, mgmt: {}us, spikes: {}us."\
                .format(np.round(time_probe.totalExecutionTime/1000000,3), time_probe.totalHostTime/1000000, 
                        int(time_probe.totalManagementTime), total_spike_time))
            print("total energy spent: {}uJ, energy spent on spiking phase: {}uJ".format(int(energy_probe.totalEnergy), int(total_spike_energy)))
            assert time_probe.timeUnits == 'us'
            assert energy_probe.energyUnits == 'uJ'
            return total_spike_time, total_spike_energy
        else:
            # remove empty probes
            probe_dict = {}
            for name, graphs in name_probes_dicts.items():
                sums = [sum(graph.data) for graph in graphs]
                if sum(sums) != 0: probe_dict[name] = graphs

            values, spike_times = decode_spike_timings(probe_dict, t_min=self.model.t_min, t_max=t_max)

            if plot: plot_probes(probe_dict, x_limit=run_time, values=values, spike_times=spike_times, save_figure=save_plot)
            self.data['values'] = values
            self.data['spike_times'] = spike_times

    def add_spike_generators(self, spike_list, neuron_method, net, quartz_neurons, net_neurons, weight_exponent):
        input_neurons = neuron_method(promoted=True)
        if len(input_neurons) == 0: input_neurons = neuron_method() 
        if self.model.weight_e > 255:
            weight = math.ceil(self.model.weight_e / 2)
            exponent = weight_exponent + 1
        else:
            weight = self.model.weight_e
            exponent = weight_exponent
        n_inputs = 0
        for i, input_spikes in enumerate(spike_list):
            input_spike_generator = net.createSpikeGenProcess(numPorts=1)
            index_target_neuron = quartz_neurons.index(input_neurons[i])
            input_spike_generator.connect(net_neurons[index_target_neuron], 
                                          prototype=nx.ConnectionPrototype(weight=weight, weightExponent=exponent))
            input_spike_generator.addSpikes(spikeInputPortNodeIds=0, spikeTimes=input_spikes)
            n_inputs += 1
        assert len(input_neurons) == n_inputs
        return net
