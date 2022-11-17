import itertools
from torch import nn
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
import audio_samples_py as aus

import utils.utils as utils

def plot_history(error_tracker: utils.ErrorTracker, total_batches: int, eval_funcs):
    train_iter, train_log_losses, train_evals = error_tracker.train_history()
    val_iter, val_log_losses, val_evals = error_tracker.validation_history()

    num_subplots = len(eval_funcs) + 1
    assert len(train_evals) == len(eval_funcs)
    assert len(val_evals) == len(eval_funcs)
    
    display.clear_output(wait=True)
    plt.figure(figsize=(16, 4))

    plt.subplot(1, num_subplots, 1)
    plt.plot(train_iter, train_log_losses, label="Training Loss")
    plt.plot(val_iter, val_log_losses, label="Validation Loss")
    plt.xlim(0, total_batches+1)
    plt.legend()

    for i, eval_func in enumerate(eval_funcs):
        label = eval_func["label"]
        y_min,y_max = eval_func["ylim"]

        plt.subplot(1, num_subplots, 2+i)
        plt.plot(train_iter, train_evals[i], label=f"Training {label}")
        plt.plot(val_iter, val_evals[i], label=f"Validation {label}")
        plt.xlim(0, total_batches+1)
        plt.ylim(y_min, y_max)
        if "plot_misc" in eval_func:
            eval_func["plot_misc"]()
        plt.legend()
    
    plt.show()

def frequency_data(net: nn.Module, data_loader: utils.DataLoader, num_batches: int, eval_funcs):
    data = []

    for _, (signal, fft, batch_target, batch_labels) in enumerate(itertools.islice(utils.cycle_data_loader(data_loader), num_batches)):
        signal = utils.to_torch(signal)
        fft = utils.to_torch(fft)
        batch_output = net(signal, fft)

        batch_output = utils.to_numpy(batch_output)

        for i in range(signal.shape[0]):
            output = batch_output[i,:]
            target = batch_target[i,:]
            label = batch_labels[i]
            
            data_row = [label.frequency_map()]
            for eval_func in eval_funcs:
                func = eval_func["func"]
                
                data_row.append(func(output, target, label))
            data.append(data_row)
    return data

def map_to_bin(map, min_freq_map, max_freq_map, num_bins):
    map_range = max_freq_map - min_freq_map
    bin = int(np.floor((map - min_freq_map)/map_range*num_bins))
    if bin == 30:
        bin = 29
    return bin

def bin_to_map(bin, min_freq_map, max_freq_map, num_bins):
    map_range = max_freq_map - min_freq_map
    return float(bin)/num_bins*map_range+min_freq_map

def frequency_plot(net: nn.Module, data_loader: DataLoader, num_batches: int, eval_funcs, bins):
    data = frequency_data(net, data_loader, num_batches, eval_funcs)

    min_freq_map = min([row[0] for row in data])
    max_freq_map = max([row[0] for row in data])

    num_subplots = len(eval_funcs)

    display.clear_output(wait=True)
    plt.figure(figsize=(12, 4))

    for i, eval_func  in enumerate(eval_funcs):
        (y_min, y_max) = eval_func["ylim"]
        label = eval_func["label"]

        bin_sums = [0 for _ in range(bins)]
        bin_counts = [0 for _ in range(bins)]
        for row in data:
            frequency_map = row[0]
            value = row[i+1]
            bin = map_to_bin(frequency_map, min_freq_map, max_freq_map, bins)
            bin_sums[bin] += value
            bin_counts[bin] += 1
        bin_values = [bin_sums[j]/float(bin_counts[j]) if bin_counts[j] != 0 else 0 for j in range(bins)]

        bin_frequency_map = [bin_to_map(bin, min_freq_map, max_freq_map, bins) for bin in range(bins+1)]
        bin_frequency = [aus.map_to_frequency(freq_map) for freq_map in bin_frequency_map]


        plt.subplot(1, num_subplots, 1+i)
        plt.bar(bin_frequency[:-1], height=bin_values, width=np.diff(bin_frequency), align="edge", edgecolor='black', label=label)
        plt.xscale("log")
        plt.ylim(y_min, y_max)
        plt.legend()
    plt.show()