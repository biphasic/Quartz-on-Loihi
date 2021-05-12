from quartz.components import Neuron
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
import os
from collections import defaultdict 


class Network:
    """
    Network is similar to the keras Sequential class and will store all the information about layers. 
    By passing the inputs, the model will be compiled using nxSDK and run on Loihi.
    
    Args:
        t_max: the number of time steps to encode the value of 1. Higher t_max means more fine-grained resolution, but also slower execution on Loihi. Use powers of 2
        layers: a list of quartz.layer objects with respective parameters.
        
    Returns:
        model - the higher level connected model that can be used for inspection of connections and parameters. Not yet compiled for Loihi
    """
#     @profile
    def __init__(self, t_max, layers, verbose=False):
        assert np.log2(t_max).is_integer()
        self.t_max = t_max
        self.layers = layers
        if verbose:
            from tqdm.auto import tqdm
            for i in tqdm(range(1, len(self.layers)), unit='layer'):
                self.layers[i].connect_from(self.layers[i-1], t_max)
        else:
            for i in range(1, len(self.layers)):
                self.layers[i].connect_from(self.layers[i-1], t_max)
        # assign vth_mants according to t_max
        self.check_vth_mants()

    def n_output_compartments(self):
        return sum([layer.n_output_compartments() for layer in self.layers])
    
    def n_bias_compartments(self):
        return sum([layer.n_bias_compartments() for layer in self.layers])

    def n_parameters(self):
        return sum([layer.n_parameters() for layer in self.layers])
    
    def n_outgoing_connections(self):
        return sum([layer.n_outgoing_connections() for layer in self.layers])

    def check_vth_mants(self):
        for layer in self.layers:
            layer.vth_mant = 2**np.log2(self.t_max*layer.weight_acc)

    def set_probe_t_max(self):
        for layer in self.layers:
            if layer.monitor: layer.probe.t_max = self.t_max
            for block in layer.blocks:
                if block.monitor: block.probe.t_max = self.t_max

    def __call__(self, inputs, profiling=False, logging=False, partition='loihi'):
        np.set_printoptions(suppress=True)
        batch_size = inputs.shape[0] if len(inputs.shape) == 4 else 1
        # figure out presentation time for 1 sample and overall run time
        n_layers = len([layer for layer in self.layers if not isinstance(layer, quartz.layers.MaxPool2D)])
        time_add = 0.9 if (not isinstance(self.layers[-1], quartz.layers.MaxPool2D) and not self.layers[-1].rectifying) else 0
        self.steps_per_image = int((n_layers+time_add)*self.t_max + 2*n_layers + 1) # add a fraction of t_max (time_add) at the end in case of non-rectifying last layer
        run_time = self.steps_per_image*batch_size
        input_spike_list = quartz.decode_values_into_spike_input(inputs, self.t_max, self.steps_per_image) # use latency encoding
        self.data = []
        self.logging = logging
        if not logging: set_verbosity(LoggingLevel.ERROR)
        # monitor output layer and setup probes
        if not profiling: output_probe = quartz.probe(self.layers[-1])
        self.set_probe_t_max()
        # create and connect compartments and add input spikes
        board = self.build_model(input_spike_list)
        # use reset snip in case of multiple samples
        if batch_size > 1: 
            board = self.add_snips(board)
        else:
            self.init_channel = None # no snips initialisation
        if self.logging: print("{} Executing model...".format(datetime.datetime.now()))
        # execute        
        self.run_on_loihi(board, run_time, profiling, partition)
        if profiling:
            return self.energy_probe
        else:
            self.data = output_probe.output()
            output_array = output_probe.output()[1]
            last_layer = self.layers[-1]
            if isinstance(last_layer, quartz.layers.Dense):
                self.first_spikes = np.zeros((last_layer.output_dims,batch_size))
                for i, (key, values) in enumerate(sorted(self.data[0].items())[1:last_layer.output_dims+1]):
                    for iteration in range(batch_size):
                        self.first_spikes[i, iteration] = values[(values>(iteration * self.steps_per_image)) & (values<((iteration+1)*self.steps_per_image))][0]
                try:
                    output_array = output_array.reshape(last_layer.output_dims, batch_size).T
                except:
                    print("Could not reshape output to desired size...")
            else:
                output_array = output_array.reshape(*last_layer.output_dims, batch_size)
                output_array = np.transpose(output_array, (3,0,1,2))
            return output_array

    def build_model(self, input_spike_list):
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
        board = nx.N2Compiler().compile(net)
        # print some tracked neurons
        for ID in self.tracker_ids:
            print(net.resourceMap.compartment(ID))
        return board

    def check_layout(self):
        self.compartments_on_core = np.zeros((128*32))
        self.biases_on_core = np.zeros((128*32))
        # since we are using axon delays for bias neurons, we need as many compartment profiles as biases in our model
        max_cx_per_core = 1024
        max_synapses_per_core = 24000 # this number should be higher
        max_incoming_axons_per_core = 4096
        max_outgoing_axons_per_core = 4096

        # evaluate how many cores are needed for each layer based on the constraints for each core
        for i, layer in enumerate(self.layers):
            if i == 0: 
                layer.n_cores = math.ceil(np.product(self.layers[0].output_dims)/max_cx_per_core)
                continue
            n_cores_cxs = len(layer.neurons()) / max_cx_per_core
            n_cores_synapses = sum([neuron.n_incoming_synapses for neuron in layer.neurons()]) / max_synapses_per_core
            n_cores_incoming_axons = sum([len(block.connections) for block in self.layers[i-1].blocks]) * self.layers[i-1].n_cores / max_incoming_axons_per_core
            layer.n_cores = math.ceil(max(n_cores_cxs, n_cores_synapses, n_cores_incoming_axons))
            layer.n_cx_per_core = math.ceil(len(layer.neurons()) / layer.n_cores)
            layer.n_bias_per_core = math.ceil(len(layer.bias_neurons) / layer.n_cores)
#             print("Layer {0:1.0f}: {2:1.1f} cores for compartments, {3:1.1f} cores for synapses, {4:1.1f} cores for incoming axons, choosing {5:1.0f}."\
#                   .format(i, n_cores_cxs, n_cores_synapses, n_cores_incoming_axons, layer.n_cores))

            # if we spread out the current layer over too many cores, then the previous layer will have a problem with the number of output axons. 
            # We'll therefore also increase the number of cores for the previous layer
            n_cores_outgoing_axons = sum([len(block.connections) for block in self.layers[i-1].blocks]) * layer.n_cores / 6 / max_outgoing_axons_per_core # division by 6 is a simple heuristic
            if n_cores_outgoing_axons > self.layers[i-1].n_cores:
                self.layers[i-1].n_cores = math.ceil(n_cores_outgoing_axons)
                self.layers[i-1].n_cx_per_core = math.ceil(len(self.layers[i-1].neurons()) / self.layers[i-1].n_cores)
                self.layers[i-1].n_bias_per_core = math.ceil(len(self.layers[i-1].bias_neurons) / self.layers[i-1].n_cores)
#                 print("Updated n_cores for previous layer due to large number of outgoing axons: " + str(self.layers[i-1].n_cores))

        self.n_cores = sum([layer.n_cores for layer in self.layers[1:]])
        
        # distribute neurons equally across number of cores per layer
        core_id = 0
        for i, layer in enumerate(self.layers[1:]):
            bias_core_id = core_id
            for neuron in layer.bias_neurons:
                if self.biases_on_core[bias_core_id] + 1 > layer.n_bias_per_core: bias_core_id += 1
                neuron.core_id = bias_core_id
                self.compartments_on_core[bias_core_id] += 1
                self.biases_on_core[bias_core_id] += 1
            for neuron in layer.neurons_without_bias():
                if self.compartments_on_core[core_id] + 1 > layer.n_cx_per_core: core_id += 1
                neuron.core_id = core_id
                self.compartments_on_core[core_id] += 1
            core_id += 1
#         if self.logging:
#             print("Number of compartments on each core for 1 chip:")
#             print(self.compartments_on_core.reshape(16,8))
        
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

        self.tracker_ids = []
        # create neurons for all subsequent layers
        for i, layer in enumerate(self.layers[1:]):
            for neuron in layer.neurons_without_bias():
                acc_proto = nx.CompartmentPrototype(logicalCoreId=neuron.core_id, vThMant=layer.vth_mant, compartmentCurrentDecay=0, tEpoch=63)
                pulse_proto = nx.CompartmentPrototype(logicalCoreId=neuron.core_id, vThMant=layer.weight_e - 1, compartmentCurrentDecay=4095, tEpoch=63)
                loihi_neuron = net.createCompartment(acc_proto) if neuron.current_type == Neuron.accumulation else net.createCompartment(pulse_proto)
#                 if "sync" in neuron.name or "rectifier" in neuron.name: self.tracker_ids.append(loihi_neuron.nodeId)
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
                if block.monitor: block.probe.set_loihi_probe(block_group.probe(full_measurements))
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
                    prototype = nx.ConnectionPrototype(weightExponent=exponent, signMode=1, compressionMode=3)
                    if isinstance(target, quartz.components.Neuron) and target.loihi_block is None: # an nxSDK compGroup can only connect to another group
                        loihi_block = net.createCompartmentGroup(size=0, name=target.name)
                        loihi_block.addCompartments(target.loihi_neuron)
                        target.loihi_block = loihi_block
                    block.loihi_block.connect(target.loihi_block, prototype=prototype, connectionMask=mask, weight=weights, delay=np.array(delays))
            for neuron in layer.neurons():
                for target, weight, exponent, delay in neuron.synapses:
                    if weight != 0:
                        if neuron.type == Neuron.bias: delay = 0 # axon delay already defined in cx prototype
                        prototype = nx.ConnectionPrototype(weightExponent=exponent, weight=np.array(weight), delay=np.array(delay), signMode=1)
                        neuron.loihi_neuron.connect(target.loihi_neuron, prototype=prototype)
                neuron.loihi_block = None # reset for next run with same model
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
        if self.init_channel is not None:
            self.init_channel.write(1, [self.steps_per_image])
            self.init_channel.write(1, [self.n_cores])
        board.run(run_time)
        board.disconnect()
        if profiling:
            self.power_stats = board.energyTimeMonitor.powerProfileStats
            print(self.power_stats)

    def __repr__(self):
        print("layer name   \t  n_comp n_bias  n_param    n_conn")
        print("---------------------------------------------------")
        print('\n'.join(["{:11s}\t{:8d} {:8d} {:7d} {:9d}".format(layer.name, layer.n_output_compartments(), layer.n_bias_compartments(), layer.n_parameters(),
                                                         layer.n_outgoing_connections()) for layer in self.layers]))
        print("---------------------------------------------------")
        return "{:11s}\t{:8d} {:8d} {:7d} {:9d}".format("total", self.n_output_compartments(), self.n_bias_compartments(), self.n_parameters(), self.n_outgoing_connections())

