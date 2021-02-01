from quartz.neuron import Neuron
from quartz.utils import decode_spike_timings, profile
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
from tqdm.auto import tqdm


class Network:
    def __init__(self, t_max, layers, name=''):
        self.name = name
        assert np.log2(t_max).is_integer()
        self.t_max = t_max
        self.layers = layers
        self.probes = []
        if len(self.layers) > 5:
            for i in tqdm(range(1, len(self.layers)), unit='layer'): # skip first layer
                self.layers[i].connect_from(self.layers[i-1], t_max)
        else:
            for i in range(1, len(self.layers)): # skip first layer
                self.layers[i].connect_from(self.layers[i-1], t_max)
        # assign vth_mants according to t_max
        self.check_vth_mants()
        
    def __call__(self, inputs, profiling=False, logging=False, partition='loihi'):
        np.set_printoptions(suppress=True)
        batch_size = inputs.shape[0] if len(inputs.shape) == 4 else 1
        # figure out presentation time for 1 sample and overall run time
        n_layers = len([layer for layer in self.layers if not isinstance(layer, quartz.layers.MaxPool2D)])
        self.steps_per_image = int((n_layers+0.9)*self.t_max)
        run_time = self.steps_per_image*batch_size
        input_spike_list = quartz.decode_values_into_spike_input(inputs, self.t_max, self.steps_per_image)
        self.data = []
        self.logging = logging
        if not logging: set_verbosity(LoggingLevel.ERROR)
        # monitor output layer and setup probes
        if not profiling: output_probe = quartz.probe(self.layers[-1])
        self.set_probe_t_max()
        # create and connect compartments and add input spikes
        board = self.build_model(input_spike_list)
        # use reset snip in case of multiple samples
        if batch_size > 1: board = self.add_snips(board)
        # execute
        self.run_on_loihi(board, run_time, profiling, partition)
        if profiling:
            return self.energy_probe
        else:
            self.data = output_probe.output()
            output_array = output_probe.output()[1]
            last_layer = self.layers[-1]
            if isinstance(last_layer, quartz.layers.Dense):
                try:
                    output_array = output_array.reshape(last_layer.output_dims, batch_size).T
                except:
                    pass
            else:
                output_array = output_array.reshape(*last_layer.output_dims, batch_size)
                output_array = np.transpose(output_array, (3,0,1,2))
            return output_array

    def build_model(self, input_spike_list):
        net = nx.NxNet()
        # assign core layout based on no of compartments and no of unique connections
        self.check_layout()
        # create loihi compartments
        net = self.create_compartments()
        # connect loihi compartments
        net = self.connect_blocks(net)
        # add inputs
        self.add_input_spikes(input_spike_list)
        # compile the whole thing and return board
        if self.logging: print("{} Compiling model...".format(datetime.datetime.now()))
        return nx.N2Compiler().compile(net)

    def n_output_compartments(self):
        return sum([layer.n_output_compartments() for layer in self.layers])
    
    def n_bias_compartments(self):
        return sum([layer.n_bias_compartments() for layer in self.layers])

    def n_parameters(self):
        return sum([layer.n_parameters() for layer in self.layers])
    
    def n_outgoing_connections(self):
        return sum([layer.n_outgoing_connections() for layer in self.layers])
    
    def n_recurrent_connections(self):
        return sum([layer.n_recurrent_connections() for layer in self.layers])

    def check_vth_mants(self):
        for layer in self.layers:
            layer.vth_mant = 2**np.log2(self.t_max*layer.weight_acc)

    def set_probe_t_max(self):
        for layer in self.layers:
            if layer.monitor: layer.probe.t_max = self.t_max
            for block in layer.blocks:
                if block.monitor: block.probe.t_max = self.t_max

    def check_layout(self):
        self.core_ids = np.zeros((128))
        self.compartments_on_core = np.zeros((128))
        self.compartments_on_core = np.zeros((128))
        n_compartments_per_core = 1024
        n_incoming_axons_per_core = 4096
        n_outgoing_axons_per_core = 4096
        core_id = 0
        for i, layer in enumerate(self.layers[1:]):
            max_comps_per_core = 130
            for neuron in layer.neurons():
                if core_id >= 127: 
                    print(self.core_ids)
                    print(self.compartments_on_core)
                    raise NotImplementedError("Too many neurons for one Loihi chip")
                if self.compartments_on_core[core_id] + 1 >= max_comps_per_core:
                    core_id += 1
                self.core_ids[core_id] = i
                neuron.core_id = core_id
                self.compartments_on_core[core_id] += 1
            core_id += 1
        
    def create_compartments(self):
        net = nx.NxNet()
        full_measurements = [nx.ProbeParameter.SPIKE, nx.ProbeParameter.COMPARTMENT_VOLTAGE, nx.ProbeParameter.COMPARTMENT_CURRENT]
        spike_counter = [nx.ProbeParameter.SPIKE]

        # input layer uses spike generators instead of neurons
        self.input_spike_gen = net.createSpikeGenProcess(numPorts=len(self.layers[0].output_neurons))
        self.layers[0].sync_neurons[0].loihi_neuron = net.createSpikeGenProcess(numPorts=len(self.layers[0].sync_neurons))
        self.layers[0].rectifier_neurons[0].loihi_neuron = net.createSpikeGenProcess(numPorts=len(self.layers[0].rectifier_neurons))
        for i, neuron in enumerate(self.layers[0].output_neurons):
            neuron.loihi_neuron = self.input_spike_gen.spikeInputPortGroup.nodeSet[i]
        for block in self.layers[0].blocks:
            block_group = net.createSpikeInputPortGroup(size=0, name=block.name)
            for neuron in block.neurons:
                block_group.addSpikeInputPort(neuron.loihi_neuron)
            block.loihi_block = block_group

        # other layers
        for i, layer in enumerate(self.layers[1:]):
            for neuron in layer.neurons_without_bias():
                acc_proto = nx.CompartmentPrototype(logicalCoreId=neuron.core_id, vThMant=layer.vth_mant, compartmentCurrentDecay=0, tEpoch=63)
                pulse_proto = nx.CompartmentPrototype(logicalCoreId=neuron.core_id, vThMant=layer.weight_e - 1, compartmentCurrentDecay=4095, tEpoch=63)
                loihi_neuron = net.createCompartment(acc_proto) if neuron.loihi_type == Neuron.acc else net.createCompartment(pulse_proto)
                neuron.loihi_neuron = loihi_neuron
                if neuron.monitor: neuron.probe.set_loihi_probe(loihi_neuron.probe(full_measurements))
            for neuron in layer.bias_neurons:
                bias_proto = nx.CompartmentPrototype(logicalCoreId=neuron.core_id, vThMant=layer.weight_e - 1, 
                                                     compartmentCurrentDecay=4095, tEpoch=63, axonDelay=neuron.synapses[0][3])
                loihi_neuron = net.createCompartment(bias_proto)
                neuron.loihi_neuron = loihi_neuron
            for block in layer.blocks:
                block_group = net.createCompartmentGroup(size=0, name=block.name)
                block_group.addCompartments([neuron.loihi_neuron for neuron in block.neurons])
                block.loihi_block = block_group
                if block.monitor:
                    block.probe.set_loihi_probe(block_group.probe(full_measurements))
            if layer.monitor:
                layer.probe.set_loihi_probe([neuron.loihi_neuron.probe(spike_counter)[0] for neuron in layer.neurons_without_bias()])
        return net

#     @profile # uncomment to profile model building
    def connect_blocks(self, net):
        if self.logging: print("{} Loihi neuron creation done, now connecting...".format(datetime.datetime.now()))
        for l, layer in enumerate(self.layers):
            for block in layer.blocks:
                for target, weights, exponent, delays in block.connections:
                    mask = np.array(weights != 0)
                    conn_prototypes = [nx.ConnectionPrototype(weightExponent=exponent, signMode=2),
                                       nx.ConnectionPrototype(weightExponent=exponent, signMode=3),]
                    proto_map = np.zeros_like(weights).astype(int)
                    proto_map[weights<0] = 1
                    weights = weights.round()
                    if np.sum(proto_map[proto_map==mask]) == np.sum(mask): # fixes issue when only prototype[1] (negative conns) is used in connections
                        conn_prototypes[0] = conn_prototypes[1]
                        proto_map = np.zeros_like(weights).astype(int)
                    if len(block.neurons) == 1: # fixes issue with broadcasting the connectionMask when there is only 1 source neuron
                        mask = mask.reshape(-1, 1)
                    if isinstance(target, quartz.neuron.Neuron) and not hasattr(target, 'loihi_block'): # an nxSDK compGroup can only connect to another group
                        loihi_block = net.createCompartmentGroup(size=0, name=target.name)
                        loihi_block.addCompartments(target.loihi_neuron)
                        target.loihi_block = loihi_block
                    block.loihi_block.connect(target.loihi_block, prototype=conn_prototypes, connectionMask=mask,
                                              prototypeMap=proto_map, weight=weights, delay=np.array(delays))
            for neuron in layer.neurons_without_bias():
                for target, weight, exponent, delay in neuron.synapses:
                    if weight != 0:
                        prototype = nx.ConnectionPrototype(weightExponent=exponent, weight=np.array(weight), delay=np.array(delay), signMode=2 if weight >= 0 else 3)
                        neuron.loihi_neuron.connect(target.loihi_neuron, prototype=prototype)
            for neuron in layer.bias_neurons:
                for target, weight, exponent, delay in neuron.synapses:
                    if weight != 0:
                        prototype = nx.ConnectionPrototype(weightExponent=exponent, weight=np.array(weight), delay=0, signMode=2 if weight >= 0 else 3)
                        neuron.loihi_neuron.connect(target.loihi_neuron, prototype=prototype)
        return net

    def add_input_spikes(self, spike_list):
        for s, spikes in enumerate(spike_list):
            if s == 0:
                self.layers[0].sync_neurons[0].loihi_neuron.addSpikes(spikeInputPortNodeIds=0, spikeTimes=spikes)
            elif s == 1:
                self.layers[0].rectifier_neurons[0].loihi_neuron.addSpikes(spikeInputPortNodeIds=0, spikeTimes=spikes)
            else:
                self.input_spike_gen.addSpikes(spikeInputPortNodeIds=s-2, spikeTimes=spikes)

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
        board.start(partition=partition)
        try:
            self.init_channel.write(1, [self.steps_per_image])
        except:
            pass # raise NotImplementedError()
        board.run(run_time)
        board.disconnect()
        if profiling:
            self.power_stats = board.energyTimeMonitor.powerProfileStats
            print(self.power_stats)

    def print_core_layout(self, redo=True):
        if redo: self.check_layout()
        print(self.core_ids)
        print(self.compartments_on_core)

    def __repr__(self):
        print("layer name   \t  n_comp n_bias  n_param    n_conn")
        print("---------------------------------------------------")
        print('\n'.join(["{:11s}\t{:8d} {:8d} {:7d} {:9d}".format(layer.name, layer.n_output_compartments(), layer.n_bias_compartments(), layer.n_parameters(),
                                                         layer.n_outgoing_connections()) for layer in self.layers]))
        print("---------------------------------------------------")
        return "{:11s}\t{:8d} {:8d} {:7d} {:9d}".format("total", self.n_output_compartments(), self.n_bias_compartments(), self.n_parameters(), self.n_outgoing_connections())

