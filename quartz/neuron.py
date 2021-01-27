class Neuron:
    input, output, bias, sync, rectifier = range(5)
    pulse, acc = range(2)

    def __init__(self, type=output, loihi_type=pulse, name=None, monitor=False):
        self.type = type
        self.loihi_type = loihi_type
        self.name = name
        self.monitor = monitor
        self.synapses = []
        self.connections = []

    def connect_to(self, target_neuron, weight, exponent=0, delay=0):
        self.synapses.append((target_neuron, weight, exponent, delay))
        
    def group_connect_to(self, target_group, weight, exponent=0, delay=0):
        self.connections.append((target_group, weight, exponent, delay))

    def __repr__(self):
        return self.name
