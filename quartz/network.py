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
        
    def __call__(self, input_spike_list, t_max, num_chips=1):
        board, probes = self.build_model(input_spike_list, t_max)
        self.run_on_loihi(board, probes)

    def build_model(self, input_spike_list, t_max):
        net = nx.NxNet()
        vth_mant=2**16
        assert np.log2(t_max).is_integer()
        # add intermediate neurons for delay encoder depending on t_max
        self.check_block_delays(t_max, 2**3)
        # assign weight exponents depending on t_max
        self.check_weight_exponents(t_max, vth_mant)
        # assign core layout based on no of compartments and no of unique connections
        self.check_layout()
        # create loihi compartments
        net = self.create_compartments(vth_mant)
        # connect loihi compartments
        self.connect_compartments(t_max, net)
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

    def check_weight_exponents(self, t_max, vth_mant):
        for layer in self.layers:
            layer.weight_exponent = np.log2(vth_mant/(t_max*layer.weight_acc))
        
    def check_layout(self):
        self.core_ids = np.zeros((128))
#         n_compartments = [layer.n_compartments() for layer in self.layers]
#         n_connections = [layer.n_connections() for layer in self.layers]
#         n_parameters = [layer.n_parameters() for layer in self.layers]
        core_id = 0
        compartments_on_core = np.zeros((128))
        for i, layer in enumerate(self.layers):
            self.core_ids[core_id] = i
            for block in layer.blocks:
                if compartments_on_core[core_id] + len(block.neurons) >= 1024:
                    core_id += 1
                    self.core_ids[core_id] = i
                block.core_id = core_id
                compartments_on_core[core_id] += len(block.neurons)
            core_id += 1
        print(self.core_ids)
        print(compartments_on_core)
        
    def create_compartments(self, vth_mant):
        net = nx.NxNet()
        for i, layer in enumerate(self.layers):
            for block in layer.blocks:
                block_group = net.createCompartmentGroup(size=0)
                layer.compartment_groups.append(block_group)
                acc_proto = nx.CompartmentPrototype(logicalCoreId=block.core_id, vThMant=vth_mant, compartmentCurrentDecay=0)
                for neuron in block.neurons:
                    if neuron.loihi_type == Neuron.acc:
                        block_group.addCompartments(net.createCompartment(acc_proto))
                    else:
                        pulse_mant = (layer.weight_e - 1) * 2**layer.weight_exponent
                        no_inputs = len(neuron.incoming_synapses())
                        if no_inputs > 0:
                            assert no_inputs <= layer.weight_e
                            ratio = (layer.weight_e // no_inputs) * no_inputs / layer.weight_e
                            if ratio != 1:
                                pulse_mant = layer.weight_e * ratio * 2**layer.weight_exponent - 1
                        pulse_proto = nx.CompartmentPrototype(logicalCoreId=block.core_id, vThMant=pulse_mant, compartmentCurrentDecay=4095)
                        block_group.addCompartments(net.createCompartment(pulse_proto))
                block_group.connect(block_group, weight=block.internal_weight_matrix(), delay=block.internal_delay_matrix(), connectionMask=block.internal_connection_matrix())
        #ipdb.set_trace()
        return net
#                 relcos = layer.get_relco_blocks()

    def connect_compartments(self, t_max, net):
        for i in range(1,len(self.layers)):
            layer = self.layers[i]
            for blocks in layer:
                ipdb.set_trace()
        
        
    def run_on_loihi(self, board, probes):
        pass
        

    def __repr__(self):
        print("name     \tn_comp \tn_param n_conn")
        print("-------------------------------------")
        print('\n'.join(["{}    \t{}  \t{}  \t{}".format(layer.name, layer.n_compartments(), layer.n_parameters(),
                                                         layer.n_connections()) for layer in self.layers]))
        print("-------------------------------------")
        return "total   \t{}  \t{}  \t{}".format(self.n_compartments(), self.n_parameters(), self.n_connections())
    
    
    #@profile
    def run_on_loihi1(self, run_time, input_spike_list=None, recall_spike_list=None, t_max=None, vth_mant=2**16, full_probes=True, num_chips=1,
                     partition='loihi', probe_selection=[], probe_interval=1, profiling=False, plot=True, logging=False, save_plot=False):
        board, probes = self.build_loihi_model(input_spike_list, recall_spike_list, t_max, vth_mant, logging,
                                               full_probes, num_chips, probe_selection, probe_interval, profiling)
        return self.run_model(board, probes, run_time, t_max, profiling, partition, plot, save_plot)
        
    def build_loihi_model(self, input_spike_list=None, recall_spike_list=None, t_max=None, vth_mant=2**16, logging=False, full_probes=True, 
                          num_chips=1, probe_selection=[], probe_interval=1, profiling=False):

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
