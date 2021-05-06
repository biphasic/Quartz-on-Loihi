import matplotlib.pyplot as plt
import numpy as np
import collections
import cProfile, pstats, io


def extract_spike_timings(probe_dict):
    spike_times = {}
    for index, (name, graphs) in enumerate(probe_dict.items()):
        spike_times_binary = np.array(graphs.data)
        spike_time_indices = np.where(spike_times_binary == 1)[0]
        if len(spike_time_indices) > 0: spike_times[name] = spike_time_indices
    return spike_times

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

def decode_values_into_spike_input(samples, t_max, steps_per_sample, t_min=1, start_time=2):
    assert samples.max() <= 1 and samples.min() >= 0
    while len(samples.shape) < 4:
        samples = np.expand_dims(samples, axis=0)
    if samples.shape[0] > 1 and steps_per_sample == 0: raise Exception("Specify # of steps per sample")
    inputs = [[] for i in range(int(np.product(samples.shape[1:])))]
    inputs.append([])
    inputs.append([])
    for sample in samples:
        inputs[0] += [int(start_time)] # sync
        inputs[1] += [int(start_time+t_max-t_min)] # rectifier
        for c, channel in enumerate(sample):
            for i, value in enumerate(channel.flatten()):
                inputs[c*len(channel.flatten())+i+2] += [int((t_max*(1-value)).round() + start_time)]
        start_time += steps_per_sample
    return inputs

def quantize_parameters(weights, biases, weight_acc, t_max):
    quantized_weights = (weight_acc*weights).round()/weight_acc
    quantized_biases = (biases*t_max).round()/t_max
    return quantized_weights, quantized_biases

def quantize_inputs(inputs, t_max):
    return (inputs*t_max).round()/t_max

def profile(fnc):
    """A decorator that uses cProfile to profile a function"""
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval
    return inner