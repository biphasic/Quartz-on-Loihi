class Block:
    input, output, bias, sync, rectifier = range(5)

    def __init__(self, neurons, name=None, monitor=False):
        self.type = type
        self.name = name
        self.monitor = monitor
        self.neurons = neurons
        self.connections = []

    def connect_to(self, target_group, weight, exponent=0, delay=0):
        self.connections.append((target_group, weight, exponent, delay))

    def __repr__(self):
        return self.name
