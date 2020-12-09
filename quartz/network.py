from quartz.components import Neuron, Synapse
from quartz.utils import decode_spike_timings
import quartz
import numpy as np
import math
import nxsdk
import nxsdk.api.n2a as nx
from nxsdk.graph.monitor.probes import PerformanceProbeCondition, IntervalProbeCondition, SpikeProbeCondition
from nxsdk.logutils.nxlogging import set_verbosity, LoggingLevel
from nxsdk.graph.processes.phase_enums import Phase
from nxsdk.api.enums.api_enums import ProbeParameter
from nxsdk.graph.monitor.probes import PerformanceProbeCondition
import datetime
import ipdb
import os


class Network:
    def __init__(self, layers, name=''):
        self.name = name
        self.layers = layers
        self.probes = []
        self.layout_complete = False
        for i in range(1, len(self.layers)): # skip first layer
            self.layers[i].connect_from(self.layers[i-1])
        
    def __call__(self, inputs, t_max, steps_per_image=None, profiling=False, logging=False, partition='loihi'):
        batch_size = inputs.shape[0] if len(inputs.shape) == 4 else 1
        if steps_per_image == None: steps_per_image = int(len(self.layers)*2*t_max)
        run_time = steps_per_image*batch_size
        input_spike_list = quartz.decode_values_into_spike_input(inputs, t_max, steps_per_image)
        print(input_spike_list)
        assert np.log2(t_max).is_integer()
        self.data = []
        self.t_max = t_max
        self.steps_per_image = steps_per_image
        if not logging:
            set_verbosity(LoggingLevel.ERROR)
        # monitor output layer and setup probes
        if not profiling:
            output_probe = quartz.probe(self.layers[-1])
        self.set_probe_t_max()
        # create and connect compartments and add input spikes
        board = self.build_model(input_spike_list)
        # use reset snip in case of multiple samples
        if batch_size > 1:
            board = self.add_snips(board)
        # execute
        self.run_on_loihi(board, run_time, profiling, partition)
        if profiling:
            return self.energy_probe
        else:
            self.data = output_probe.output()
            # print("Last timestep is " + str(np.max([np.max(value) for (key, value) in sorted(output_probe.output()[1].items())])))
            output_array = np.array([value for (key, value) in sorted(output_probe.output()[0].items())]).flatten()
            last_layer = self.layers[-2]
            if isinstance(last_layer, quartz.layers.Dense):
                if last_layer.rectifying: # False: # 
                    output_array = output_array.reshape(last_layer.output_dims, batch_size).T
            else: # if isinstance(last_layer, quartz.layers.Conv2D):
                output_array = output_array.reshape(*last_layer.output_dims, batch_size)
                output_array = np.transpose(output_array, (3,0,1,2))
            return output_array
    
    def build_model(self, input_spike_list):
        net = nx.NxNet()
        # add intermediate neurons for delay encoder depending on # dendritic delays
        self.check_block_delays(2**3)
        # set weight exponents
        self.set_weight_exponents(0)
        # assign vth_mants according to t_max
        self.check_vth_mants()
        # assign core layout based on no of compartments and no of unique connections
        self.check_layout()
        # create loihi compartments
        net = self.create_compartments()
        # connect loihi compartments
        net = self.connect_blocks(net)
        # add inputs
        self.add_input_spikes(input_spike_list)
        # compile the whole thing
        board = self.compile_net(net)
        return board

    def n_compartments(self):
        return sum([layer.n_compartments() for layer in self.layers])
    
    def n_parameters(self):
        return sum([layer.n_parameters() for layer in self.layers])
    
    def n_outgoing_connections(self):
        return sum([layer.n_outgoing_connections() for layer in self.layers])
    
    def n_recurrent_connections(self):
        return sum([layer.n_recurrent_connections() for layer in self.layers])
    
    def check_block_delays(self, numDendriticAccumulators):
        for layer in self.layers:
            layer.check_block_delays(self.t_max, numDendriticAccumulators)

    def check_vth_mants(self):
        for layer in self.layers:
            layer.vth_mant = 2**(layer.weight_exponent + np.log2(self.t_max*layer.weight_acc))

    def set_weight_exponents(self, expo):
        for layer in self.layers:
            layer.weight_exponent = expo

    def set_probe_t_max(self):
        for layer in self.layers:
            if layer.monitor: layer.probe.t_max = self.t_max
            for block in layer.blocks:
                if block.monitor: block.probe.t_max = self.t_max

    def check_layout(self):
        self.core_ids = np.zeros((128))
        self.compartments_on_core = np.zeros((128))
        n_compartments_per_core = 1024
        n_incoming_axons_per_core = 4096
        n_outgoing_axons_per_core = 4096
        core_id = 0
        for i, layer in enumerate(self.layers):
#             comps = layer.n_compartments()
#             comp_limit = comps / n_compartments_per_core
#             if i == 0:
#                 conns_in = 1 # layer.n_recurrent_connections()
#             else:
#                 conns_in = self.layers[i-1].n_outgoing_connections()
#             conns_out =  layer.n_outgoing_connections()
#             conn_in_limit = conns_in / n_incoming_axons_per_core
#             conn_out_limit = conns_out / n_outgoing_axons_per_core
#             n_cores = max(comp_limit, conn_in_limit, conn_out_limit)
#             max_comps_per_core = comps / n_cores
            if i == 0:
                max_comps_per_core = 350
            elif i == 1:
                max_comps_per_core = 150
            elif i == 2:
                max_comps_per_core = 150
            else:
                max_comps_per_core = 220
            for block in layer.blocks:
                if core_id >= 127: 
                    print(self.core_ids)
                    print(self.compartments_on_core)
                    raise NotImplementedError("Too many neurons for one Loihi chip")
                if self.compartments_on_core[core_id] + len(block.neurons) >= max_comps_per_core:
                    core_id += 1
                self.core_ids[core_id] = i
                block.core_id = core_id
                self.compartments_on_core[core_id] += len(block.neurons)
            core_id += 1
        self.layout_complete = True
        
    def create_compartments(self):
        net = nx.NxNet()
        self.spike_gen_processes = []
        measurements = [nx.ProbeParameter.SPIKE, nx.ProbeParameter.COMPARTMENT_VOLTAGE, nx.ProbeParameter.COMPARTMENT_CURRENT]
        layer_measurements = [nx.ProbeParameter.SPIKE]
        for i, layer in enumerate(self.layers):
            for block in layer.blocks:
                if isinstance(block, quartz.blocks.Input) or (i == 0 and isinstance(block, quartz.blocks.Trigger)):
                    #ipdb.set_trace()
                    block_group = net.createSpikeGenProcess(numPorts=1)
                    block.loihi_group = block_group
                    self.spike_gen_processes.append(block_group)
                    continue
                block_group = net.createCompartmentGroup(size=0, name=block.name)
                block.loihi_group = block_group
                acc_proto = nx.CompartmentPrototype(logicalCoreId=block.core_id, vThMant=layer.vth_mant, compartmentCurrentDecay=0)
                acc_proto_scaled = nx.CompartmentPrototype(logicalCoreId=block.core_id, vThMant=layer.vth_mant * layer.weight_scaling,
                                                           compartmentCurrentDecay=0) # for calc neurons
                for neuron in block.neurons:
                    if neuron.loihi_type == Neuron.acc and "calc" in neuron.name:
                        loihi_neuron = net.createCompartment(acc_proto_scaled) # increase the dynamic range of input synapse weights
                    elif neuron.loihi_type == Neuron.acc:
                        loihi_neuron = net.createCompartment(acc_proto)
                    else:
                        pulse_mant = (layer.weight_e - 1) * 2**layer.weight_exponent - 1
                        no_inputs = len(neuron.incoming_synapses())
                        if no_inputs > 0:
                            assert no_inputs <= layer.weight_e
                            ratio = (layer.weight_e // no_inputs) * no_inputs / layer.weight_e
                            if ratio != 1:
                                pulse_mant = layer.weight_e * ratio * 2**layer.weight_exponent - 1
                        pulse_proto = nx.CompartmentPrototype(logicalCoreId=block.core_id, vThMant=pulse_mant, compartmentCurrentDecay=4095)
                        loihi_neuron = net.createCompartment(pulse_proto)
                    neuron.loihi_neuron = loihi_neuron
                    block_group.addCompartments(loihi_neuron)
                    if neuron.monitor: neuron.probe.set_loihi_probe(loihi_neuron.probe(measurements))
                if block.monitor: block.probe.set_loihi_probe(block_group.probe(measurements))
            if layer.monitor:
                layer.probe.set_loihi_probe([block.loihi_group.probe(layer_measurements)[0] for block in layer.blocks])
        return net

    def connect_blocks(self, net):
        print("{} Loihi neuron creation done, now connecting...".format(datetime.datetime.now()))
        for l, layer in enumerate(self.layers):
            for target in layer.blocks:
                target_block = target.loihi_group
                for source in target.get_connected_blocks():
                    conn_prototypes = [nx.ConnectionPrototype(weightExponent=layer.weight_exponent, signMode=2),
                                       nx.ConnectionPrototype(weightExponent=layer.weight_exponent, signMode=3),]
                    source_block = source.loihi_group
                    weights, delays, mask = source.get_connection_matrices_to(target)
                    proto_map = np.zeros_like(weights).astype(int)
                    proto_map[weights<0] = 1
                    weights = weights.round()

                    if np.sum(proto_map[proto_map==mask]) == np.sum(mask): # edge case where only negative connections and conn_prototypes[0] is unused
                        conn_prototypes[0] = conn_prototypes[1]
                        proto_map = np.zeros_like(weights).astype(int)
                    if not (isinstance(target, quartz.blocks.Input) or (l == 0 and isinstance(target, quartz.blocks.Trigger))):
                        connection = source_block.connect(target_block, prototype=conn_prototypes, prototypeMap=proto_map,
                                                          weight=weights, delay=delays, connectionMask=mask)
        return net

    def add_input_spikes(self, spike_list):
        input_layer = self.layers[0]
        for spike_generator, spikes in zip(self.spike_gen_processes, spike_list):
            spike_generator.addSpikes(spikeInputPortNodeIds=0, spikeTimes=spikes)

    def compile_net(self, net):
        print("{} Compiling now...".format(datetime.datetime.now()))
        return nx.N2Compiler().compile(net) # return board
        
    def add_snips(self, board):
        snip_dir = os.getcwd() + "/quartz/snips"
        
        init_snip = board.createSnip(
            name='init-reset',
            includeDir=snip_dir,
            cFilePath=snip_dir + "/init.c",
            funcName='set_init_values',
            guardName=None,
            phase=Phase.EMBEDDED_INIT)
        
        reset_snip = board.createSnip(
            name="batch-reset",
            includeDir=snip_dir,
            cFilePath=snip_dir + "/reset.c",
            funcName="reset",
            guardName="doReset", 
            phase=Phase.EMBEDDED_MGMT)
        
        self.init_channel = board.createChannel(name=b'init_channel', elementType="int", numElements=1)
        self.init_channel.connect(None, init_snip)
        return board
    
    def run_on_loihi(self, board, run_time, profiling, partition):
        if profiling:
            pc = PerformanceProbeCondition(tStart=1, tEnd=run_time, bufferSize=2048, binSize=self.steps_per_image)
            eProbe = board.probe(ProbeParameter.ENERGY, pc)
            self.energy_probe = eProbe
        board.start()
        try:
            self.init_channel.write(1, [self.steps_per_image])
        except:
            pass
        board.run(run_time, partition=partition)
        board.disconnect()
        if profiling:
            self.power_stats = board.energyTimeMonitor.powerProfileStats
            print(self.power_stats)

    def print_core_layout(self, redo=True):
        if not self.layout_complete or redo: self.check_layout()
        print(self.core_ids)
        print(self.compartments_on_core)

    def __repr__(self):
        print("name     \tn_comp \tn_param n_conn")
        print("-------------------------------------")
        print('\n'.join(["{:11s}\t{:5d}\t{:6d}\t{:6d}".format(layer.name, layer.n_compartments(), layer.n_parameters(),
                                                         layer.n_outgoing_connections()) for layer in self.layers]))
        print("-------------------------------------")
        return "total   \t{}  \t{}  \t{}".format(self.n_compartments(), self.n_parameters(), self.n_outgoing_connections())
