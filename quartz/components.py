class Synapse:
    v, ge, gf, gate = range(4)

    def __init__(self, pre, post, weight, delay=0, type=v):
        self.type = type
        self.name = "{0} --({1};{2})-->\t{3}".format(
            pre.name, round(weight,2), delay, post.name
        )
        self.pre = pre
        self.post = post
        self.weight = weight
        self.delay = delay

    def __repr__(self):
        return self.name


class Neuron:
    input, recall, hidden, ready, output = range(5)
    pulse, acc = range(2)

    def __init__(self, type=hidden, promoted=False, loihi_type=None, name=None, monitor=False):
        self.type = type
        self.loihi_type = loihi_type
        self.promoted = promoted
        self.name = name
        self.monitor = monitor
        self.synapses = {"pre": [], "post": []}

    def connect_to(self, target_neuron, weight, delay=0, type=Synapse.v):
        self.synapses["pre"].append(
            Synapse(pre=self, post=target_neuron, weight=weight, delay=delay, type=type)
        )
        target_neuron.connect_from(self, weight, delay, type)

    def connect_from(self, source_neuron, weight, delay, type):
        self.synapses["post"].append(
            Synapse(pre=source_neuron, post=self, weight=weight, delay=delay, type=type)
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
