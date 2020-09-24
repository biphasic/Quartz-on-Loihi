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
        self.probes = []
        for i in range(1, len(layers)):
            layers[i].connect_from(layers[i-1])
        
    def __call__(self, input_spike_list, t_max):
        board, probes = self.build_model(input_spike_list, t_max)
        self.run_on_loihi(board, t_max)
        return probes

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
        net, probes = self.create_compartments(vth_mant)
        # connect loihi compartments
        net = self.connect_blocks(t_max, net)
        # add inputs
        net = self.add_input_spikes(input_spike_list, net)
        # compile the whole thing
        board = self.compile_net(net)
        return board, probes

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
#         print(self.core_ids)
#         print(compartments_on_core)
        
    def create_compartments(self, vth_mant):
        net = nx.NxNet()
        probes = []
        for i, layer in enumerate(self.layers):
            layer.compartment_groups = []
            connection_prototype = nx.ConnectionPrototype(weightExponent=layer.weight_exponent)
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
                weight, delay, mask = block.internal_connection_matrices()
                block_group.connect(block_group, prototype=connection_prototype, weight=weight, delay=delay, connectionMask=mask)
                if block.monitor: probes.append(block_group.probe([nx.ProbeParameter.SPIKE]))
        return net, probes

    def connect_blocks(self, t_max, net):
        for l, layer in enumerate(self.layers):
            if l == len(self.layers)-1: break
            connection_prototype = nx.ConnectionPrototype(weightExponent=layer.weight_exponent)
            for block in layer.blocks:
                source_block = layer.compartment_groups[layer.blocks.index(block)]
                for connection in block.connections["pre"]:
                    index_target_block = self.layers[l+1].blocks.index(connection.post)
                    target_block = self.layers[l+1].compartment_groups[index_target_block]
                    weight, delay, mask = connection.get_matrices()
                    source_block.connect(target_block, prototype=connection_prototype, weight=weight, delay=delay, connectionMask=mask)
        return net

    def add_input_spikes(self, spike_list, net):
        input_layer = self.layers[0]
        if input_layer.weight_e > 255:
            weight_e = math.ceil(self.weight_e / 2)
            weight_exponent = input_layer.weight_exponent + 1
        else:
            weight_e = input_layer.weight_e
            weight_exponent = input_layer.weight_exponent
        connection_prototype=nx.ConnectionPrototype(weightExponent=weight_exponent)
        n_inputs = 0
        for i, input_spikes in enumerate(spike_list):
            input_spike_generator = net.createSpikeGenProcess(numPorts=1)
            target_block = input_layer.compartment_groups[i] # later change this to index the right target block
            input_spike_generator.connect(target_block, prototype=connection_prototype, 
                                          weight=np.array([[weight_e],[0],[0]]),
                                          connectionMask=np.array([[1],[0],[0]]))
            input_spike_generator.addSpikes(spikeInputPortNodeIds=0, spikeTimes=input_spikes)
            n_inputs += 1
        assert len(input_layer.compartment_groups) == n_inputs
        return net

    def compile_net(self, net):
        print("{} Compiling now...".format(datetime.datetime.now()))
        return nx.N2Compiler().compile(net) # return board
        
    def run_on_loihi(self, board, t_max, partition='loihi'):
        set_verbosity(LoggingLevel.ERROR)
        run_time = len(self.layers)*6*t_max
        board.run(run_time, partition=partition)
        board.disconnect()        

    def __repr__(self):
        print("name     \tn_comp \tn_param n_conn")
        print("-------------------------------------")
        print('\n'.join(["{}    \t{}  \t{}  \t{}".format(layer.name, layer.n_compartments(), layer.n_parameters(),
                                                         layer.n_connections()) for layer in self.layers]))
        print("-------------------------------------")
        return "total   \t{}  \t{}  \t{}".format(self.n_compartments(), self.n_parameters(), self.n_connections())
