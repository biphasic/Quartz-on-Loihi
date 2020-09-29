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
        output_probe = quartz.probe(self.layers[-1])
        self.set_probe_t_max(t_max)
        board = self.build_model(input_spike_list, t_max)
        self.run_on_loihi(board, t_max)
        return np.array([value[0] for (key, value) in sorted(output_probe.output()[0].items())])
    
    def set_probe_t_max(self, t_max):
        for layer in self.layers:
            if layer.monitor: layer.probe.t_max = t_max
            for block in layer.blocks:
                if block.monitor: block.probe.t_max = t_max

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
        net = self.connect_blocks(net)
        # add inputs
        net = self.add_input_spikes(input_spike_list, net)
        # compile the whole thing
        board = self.compile_net(net)
        return board

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
            if i == 0:
                max_n_comps = 128
            else:
                max_n_comps = 512
            self.core_ids[core_id] = i
            for block in layer.blocks:
                if compartments_on_core[core_id] + len(block.neurons) >= max_n_comps:
                    core_id += 1
                    self.core_ids[core_id] = i
                block.core_id = core_id
                compartments_on_core[core_id] += len(block.neurons)
            core_id += 1
        #print(self.core_ids)
        #print(compartments_on_core)
        
    def create_compartments(self, vth_mant):
        net = nx.NxNet()
        measurements = [nx.ProbeParameter.SPIKE, nx.ProbeParameter.COMPARTMENT_VOLTAGE, nx.ProbeParameter.COMPARTMENT_CURRENT]
        layer_measurements = [nx.ProbeParameter.SPIKE]
        for i, layer in enumerate(self.layers):
            for block in layer.blocks:
                block_group = net.createCompartmentGroup(size=0)
                block.loihi_group = block_group
                acc_proto = nx.CompartmentPrototype(logicalCoreId=block.core_id, vThMant=vth_mant, compartmentCurrentDecay=0)
                for neuron in block.neurons:
                    if neuron.loihi_type == Neuron.acc:
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
            if l == len(self.layers)-1: break
            for block in layer.blocks:
                source_block = block.loihi_group
                for target in block.get_connected_blocks():
                    target_block = target.loihi_group
                    weights, delays, mask = block.get_connection_matrices_to(target)
                    if layer.weight_acc > 255:
                        conn_prototypes = [nx.ConnectionPrototype(weightExponent=layer.weight_exponent+1, signMode=2),
                                           nx.ConnectionPrototype(weightExponent=layer.weight_exponent+1, signMode=3),]
                        weights /= 2
                    else:
                        conn_prototypes = [nx.ConnectionPrototype(weightExponent=layer.weight_exponent, signMode=2),
                                           nx.ConnectionPrototype(weightExponent=layer.weight_exponent, signMode=3),]
                    proto_map = np.zeros_like(weights).astype(int)
                    proto_map[weights<0] = 1
                    weights = weights.round()
                    source_block.connect(target_block, prototype=conn_prototypes, prototypeMap=proto_map,
                                         weight=weights, delay=delays, connectionMask=mask)
                    if block == target and isinstance(block, quartz.blocks.ConstantDelay) and len(block.neurons) == 2:
                        key_delay = block.neurons[0].synapses["pre"][0].delay # connect a second time because edge case
                        delays = np.array([[0, 0],[key_delay, 0]]) # of two synapses to the same neuron with diff delays
                        source_block.connect(target_block, prototype=conn_prototypes, prototypeMap=proto_map,
                                         weight=weights, delay=delays, connectionMask=mask)
        return net

    def add_input_spikes(self, spike_list, net):
        input_layer = self.layers[0]
        if input_layer.weight_e > 255:
            weight_e = math.ceil(input_layer.weight_e / 2)
            weight_exponent = input_layer.weight_exponent + 1
        else:
            weight_e = input_layer.weight_e
            weight_exponent = input_layer.weight_exponent
        connection_prototype=nx.ConnectionPrototype(weightExponent=weight_exponent)
        n_inputs = 0
        for i, input_spikes in enumerate(spike_list):
            input_spike_generator = net.createSpikeGenProcess(numPorts=1)
            target_block = input_layer.blocks[i].loihi_group # later change this to index the right target block
            input_spike_generator.connect(target_block, prototype=connection_prototype, # change this too
                                          weight=np.array([[weight_e],[0],[0]]),
                                          connectionMask=np.array([[1],[0],[0]]))
            input_spike_generator.addSpikes(spikeInputPortNodeIds=0, spikeTimes=input_spikes)
            n_inputs += 1
        assert len(input_layer.blocks) == n_inputs
        return net

    def compile_net(self, net):
        print("{} Compiling now...".format(datetime.datetime.now()))
        return nx.N2Compiler().compile(net) # return board
        
    def run_on_loihi(self, board, t_max, partition='loihi'):
        set_verbosity(LoggingLevel.ERROR)
        run_time = len(self.layers)*3*t_max
        board.run(run_time, partition=partition)
        board.disconnect()        

    def __repr__(self):
        print("name     \tn_comp \tn_param n_conn")
        print("-------------------------------------")
        print('\n'.join(["{}    \t{}  \t{}  \t{}".format(layer.name, layer.n_compartments(), layer.n_parameters(),
                                                         layer.n_connections()) for layer in self.layers]))
        print("-------------------------------------")
        return "total   \t{}  \t{}  \t{}".format(self.n_compartments(), self.n_parameters(), self.n_connections())
