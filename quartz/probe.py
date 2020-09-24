import matplotlib.pyplot as plt
import numpy as np
import quartz
from nxsdk.utils.plotutils import plotProbes
from quartz.utils import decode_spike_timings


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
        probe_list = [probe_set.probes for probe_set in self.loihi_probe] 
        fig = plotProbes(probes = probe_list)
        #return fig
#         n_subplots = len(self.loihi_probe.probes)
#         f, axes = plt.subplots(n_subplots, 1, sharex=True)
#         for i, probe in enumerate(self.loihi_probe.probes):
#             axes[i].plot(probe.data)
            #probe.plot()
    
    def le_data(self):
        for s, probe_set in zip([self.SPIKE, self.VOLTAGE, self.CURRENT], self.loihi_probe):
            for n, probe in enumerate(probe_set):
                self.data[n].update({s: probe.data})
        return self.data
        
    def output(self):
        probes = self.loihi_probe[0].probes
        names = self.target.names()        
        return decode_spike_timings(dict(zip(names, probes)), self.t_max)
# class LayerProbe(Probe):
#     def __init__(self):
#         self.test = 3
#         pass

# class BlockProbe(Probe):
#     def __init__(self):
#         self.test = 3
#         pass