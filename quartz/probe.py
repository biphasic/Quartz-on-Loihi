import matplotlib.pyplot as plt
import numpy as np
import quartz
from nxsdk.utils.plotutils import plotProbes
from quartz.utils import extract_spike_timings, decode_spike_timings


def probe(target):
    if isinstance(target, quartz.layers.Layer):
        return LayerProbe(target)
    return NeuronProbe(target)

class Probe:
    """
    Spike probing and plotting functionality for layers or individual neurons. Number of spike counters is limited on Loihi backend.
    
    """
    SPIKE = "spikes"
    VOLTAGE = "voltage"
    CURRENT = "current"
    
    def __init__(self, target):
        self.target = target
        self.target.probe = self
        self.target.monitor = True
        self.t_max = 0
        self.data = {}
        
    def set_loihi_probe(self, probe):
        self.loihi_probe = probe
        
    def plot(self):
        fig = plotProbes(probes = self.loihi_probe)
        fig.set_size_inches((18,5))


class LayerProbe(Probe):
    def __init__(self, target):
        super(LayerProbe, self).__init__(target)

    def output(self):
        probes = self.loihi_probe
        names = self.target.names()
        names = [name for name in names]
        spike_times = extract_spike_timings(dict(zip(names, probes)))
        trigger = [value for key, value in spike_times.items() if 'rectifier' in key][0]
        outputs = [value for key, value in sorted(spike_times.items()) if 'relco' in key or 'wta' in key]
        decodings = [(trigger+1-output)/self.t_max for output in outputs]
        return spike_times, np.array(decodings)


class NeuronProbe(Probe):
    def __init__(self, target):
        super(NeuronProbe, self).__init__(target)

    def output(self):
        probes = self.loihi_probe
        names = self.target.names()
        return decode_spike_timings(dict(zip(names, probes)), self.t_max)
