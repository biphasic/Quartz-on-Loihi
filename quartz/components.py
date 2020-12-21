class Synapse:
    def __init__(self, pre, post, weight, delay=0):
        self.pre = pre
        self.post = post
        self.weight = weight
        self.delay = delay

    def __repr__(self):
        return "{0} --({1};{2})-->\t{3}".format(
            self.pre.name, round(weight,2), self.delay, self.post.name
        )


class Neuron:
    input, hidden, rectifier, output = range(4)
    pulse, acc = range(2)

    def __init__(self, type=hidden, loihi_type=pulse, name=None, monitor=False, parent=None):
        self.type = type
        self.loihi_type = loihi_type
        self.name = name
        self.monitor = monitor
        self.parent_block = parent
        self.synapses = {"pre": [], "post": []}

    def connect_to(self, target_neuron, weight, delay=0):
        self.synapses["pre"].append(
            Synapse(pre=self, post=target_neuron, weight=weight, delay=delay)
        )
#         if self != target_neuron:
#             target_neuron.connect_from(self, weight, delay)

    def connect_from(self, source_neuron, weight, delay):
        self.synapses["post"].append(
            Synapse(pre=source_neuron, post=self, weight=weight, delay=delay)
        )

    def remove_connections_to(self, target_neuron):
        for synapse in self.synapses["pre"]:
            if synapse.post == target_neuron:
                self.synapses["pre"].remove(synapse)
        target_neuron.remove_connections_from(self)

    def remove_connections_from(self, source_neuron):
        for synapse in self.synapses["post"]:
            if synapse.pre == source_neuron:
                self.synapses["post"].remove(synapse)

    def incoming_synapses(self):
        return self.synapses["post"]

    def outgoing_synapses(self):
        return self.synapses["pre"]

    def has_incoming_synapses(self):
        return self.synapses["post"] != []
    
    def reset_outgoing_connections(self):
        self.synapses["pre"] = []
    
    def reset_incoming_connections(self):
        self.synapses["post"] = []
        
    def __repr__(self):
        return self.name
