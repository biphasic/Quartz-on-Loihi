class Neuron:
    input, hidden, rectifier, output = range(4)
    pulse, acc = range(2)

    def __init__(self, type=hidden, loihi_type=pulse, name=None, monitor=False, parent=None):
        self.type = type
        self.loihi_type = loihi_type
        self.name = name
        self.monitor = monitor
        self.parent_block = parent
        self.synapses = []

    def connect_to(self, target_neuron, weight, delay=0):
        self.synapses.append((self, target_neuron, weight, delay))

    def __repr__(self):
        return self.name
