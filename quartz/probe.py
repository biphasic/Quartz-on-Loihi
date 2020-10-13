import matplotlib.pyplot as plt
import numpy as np
import quartz
import ipdb
from nxsdk.utils.plotutils import plotProbes
from quartz.utils import decode_spike_timings


def probe(target):
    if isinstance(target, quartz.layers.Layer):
        return LayerProbe(target)
    return BlockProbe(target)

class Probe:
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
    
    def le_data(self):
        for s, probe_set in zip([self.SPIKE, self.VOLTAGE, self.CURRENT], self.loihi_probe):
            for n, probe in enumerate(probe_set):
                self.data[n].update({s: probe.data})
        return self.data
    

class LayerProbe(Probe):
    def __init__(self, target):
        super(LayerProbe, self).__init__(target)

    def output(self):
        #ipdb.set_trace()
        probes = [probe.probes[0] for probe in self.loihi_probe]
        names = self.target.names()
        return decode_spike_timings(dict(zip(names, probes)), self.t_max)
    
        
class BlockProbe(Probe):
    def __init__(self, target):
        super(BlockProbe, self).__init__(target)

    def output(self):
        probes = self.loihi_probe[0].probes
        names = self.target.names()
        return decode_spike_timings(dict(zip(names, probes)), self.t_max)