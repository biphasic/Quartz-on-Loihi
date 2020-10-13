import matplotlib.pyplot as plt
import numpy as np
import ipdb
import collections


def decode_spike_timings(probe_dict, t_max, t_min=1):
    values = {}
    spike_times = {}
    for index, (name, graphs) in enumerate(probe_dict.items()):
        spike_times_binary = np.array(graphs.data)
        spike_time_indices = np.where(spike_times_binary == 1)[0]
        spike_time_interval = np.diff(spike_time_indices)[::2] # take every second interval
        value = (spike_time_interval-t_min)/(t_max)
        if len(value) > 0:
            if "-" == name[-1:]:
                values[name] = -value
            else:
                values[name] = value
        if len(spike_time_indices) > 0: spike_times[name] = spike_time_indices
    return values, spike_times

def simulate_encoded_value_accumulation(value, weight_acc, t_max=2**8, vth_mant=2**12, t_min=1):
    weight_exponent = np.log2(vth_mant/(t_max*weight_acc))
    if weight_acc/value > 255: return -1
    return np.floor(t_max**2*weight_acc**2 * 2**weight_exponent / (vth_mant*np.floor(weight_acc/value)))/t_max

def simulate_encoded_value_delay(value, t_max):
    return round(value * t_max) / t_max

def decode_values_into_spike_input(samples, t_max, steps_per_sample=0, t_min=1, start_time=1):
    assert samples.max() <= 1 and samples.min() >= 0
    if samples.shape[0] > 1 and steps_per_sample == 0: raise Error
    inputs = [[] for i in range(int(np.product(samples.shape[1:])))]
    for sample in samples:
        for channel in sample:
            for i, value in enumerate(channel.flatten()):
                inputs[i] += [start_time, (value*t_max).round()+t_min+start_time]
        start_time += steps_per_sample
    return inputs