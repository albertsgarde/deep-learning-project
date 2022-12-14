import audio_samples_py as aus
from pathlib import Path
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate, DataLoader
import numpy as np
import itertools
import matplotlib.pyplot as plt
import IPython.display as display

use_cuda = None
device = None

def setup_device(use_cuda_if_possible: bool):
    global use_cuda
    global device
    use_cuda = torch.cuda.is_available() and use_cuda_if_possible
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Running GPU.") if use_cuda else print("No GPU available.")
    return device, use_cuda

def to_numpy(x):
    """ Get numpy array for both cuda and not. """
    global use_cuda
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        if use_cuda:
            return x.cpu().data.numpy()
        return x.data.numpy()
    else:
        raise Exception(f"Unsupported type for to_numpy: {type(x)}")

def to_torch(x):
    global use_cuda
    if isinstance(x, torch.Tensor):
        if use_cuda:
            x = x.cuda()
        return x
    elif isinstance(x, np.ndarray):
        variable = Variable(torch.from_numpy(x))
        if use_cuda:
            variable = variable.cuda()
        return variable
    else:
        raise Exception(f"Unsupported type for to_torch: {type(x)}")


def custom_collate(batch):
    signals, ffts, targets, labels = list(zip(*batch))
    return default_collate(list(signals)), default_collate(list(ffts)), default_collate(list(targets)), list(labels)

class AudioRwDataSet(torch.utils.data.Dataset):
    def __init__(self, data_set: aus.DataSet, label_to_target):
         self.data_set = data_set
         self.label_to_target = label_to_target

    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, index: int):
        data_point = self.data_set[index]
        label = data_point.label()
        return data_point.samples(), data_point.audio().fft(), self.label_to_target(label), label

def init_rw_data(path, label_to_target, validation_size: float, batch_size: int):
    assert validation_size >= 0, f"validation size must be non-negative. validation_size={validation_size}"
    assert validation_size <= 1, f"validation size must be no greater than 1. validation_size={validation_size}"
    assert batch_size > 0, f"batch_size must be positive. batch_size={batch_size}"

    data_loader_params = {"batch_size": batch_size, "collate_fn": custom_collate}

    # Not a mistake. Just an artifact of how random_partition works.
    training_data, validation_data = aus.load_data_set(path).random_partition(1-validation_size)

    training_data = AudioRwDataSet(training_data, label_to_target)
    validation_data = AudioRwDataSet(validation_data, label_to_target)

    training_loader = torch.utils.data.DataLoader(training_data, **data_loader_params)
    validation_loader = torch.utils.data.DataLoader(validation_data, **data_loader_params)

    return training_loader, validation_loader

class AudioSynthDataSet(torch.utils.data.Dataset):
    def __init__(self, parameters: aus.DataParameters, label_to_target):
         self.parameters = parameters
         self.label_to_target = label_to_target

    def __len__(self):
        return np.iinfo(np.int64).max
    
    def __getitem__(self, index):
        data_point = self.parameters.generate_at_index(index)
        label = data_point.label()
        return data_point.samples(), data_point.audio().fft(), self.label_to_target(label), label

def init_synth_data(parameters: aus.DataParameters, label_to_target, seed: int, batch_size: int):
    assert seed >= 0, f"seed must be non-negative. seed={seed}"
    assert batch_size > 0, f"batch_size must be positive. batch_size={batch_size}"

    data_loader_params = {"batch_size": batch_size, "collate_fn": custom_collate}

    training_parameters = parameters.with_seed_offset(seed)
    training_loader = torch.utils.data.DataLoader(AudioSynthDataSet(training_parameters, label_to_target), **data_loader_params)
    validation_parameters = parameters.with_seed_offset(seed + 1)
    validation_loader = torch.utils.data.DataLoader(AudioSynthDataSet(validation_parameters, label_to_target), **data_loader_params)

    return training_parameters, training_loader, validation_parameters, validation_loader


    
def cycle_data_loader(data_loader):
    for _ in itertools.count(start = 0):
        generator = iter(data_loader)
        for item in generator:
            yield item

def mean_minibatch_err(output, target, label, error_function):
    assert output.shape == target.shape, f"Output and target must be same shape. Output shape: {output.shape}  target shape: {target.shape}"
    total = 0
    for i, output_row in enumerate(output):
        total += error_function(output_row, target[i,:], label[i])
    return total/output.shape[0]

def test_net(net: torch.nn.Module, validation_loader: DataLoader, criterion, num_validation_batches: int, eval_funcs):
    r"""
        Args:
            net: the model to test.
            validation_loader: a data loader that outputs validation data.
            criterion: the loss function.
            num_validation_batches: how many batches to validate the model on.
            error_functions: a list of functions taking the model output and the target and returning a floating point error measure.
    """
    was_training = net.training
    net.eval()
    total_loss = 0
    total_errors = [0] * len(eval_funcs)
    for signal, fft, target, label in itertools.islice(cycle_data_loader(validation_loader), num_validation_batches):
        signal = signal.to(device)
        fft = fft.to(device)
        target = target.to(device)
        output = net(signal, fft)
        
        total_loss += criterion(output, target).item()

        output = to_numpy(output)
        target = to_numpy(target)
        for i, eval_func in enumerate(eval_funcs):
            func = eval_func["func"]

            total_errors[i] += mean_minibatch_err(output, target, label, func)

    net.train(mode=was_training)
    return total_loss/num_validation_batches, list(map(lambda x: x / num_validation_batches, total_errors))

def manual_test(net: torch.nn.Module, validation_loader: DataLoader, num_samples: int, output_functions):
    r"""
        Args:
            net: the model to test
            validation_loader: a data loader that outputs validation data.
            num_samples: number of samples to test.
            output_functions: a map of functions taking the model output, and data point target and label as input and returning a value to be printed.
    """
    was_training = net.training
    net.eval()
    
    prints_remaining = num_samples
    for _, (signal, fft, target, label) in enumerate(validation_loader):
        signal = signal.to(device)
        fft = fft.to(device)
        output = net(signal, fft)

        output = to_numpy(output)
        for i in range(signal.shape[0]):
            if prints_remaining <= 0:
                break
            prints_remaining -= 1
            output_string = ""
            for name, output_function in output_functions.items():
                output_string += f"{name}: {output_function(output[i,:], target[i,:], label[i])} "
            print(output_string)
        if prints_remaining <= 0:
            break

    net.train(mode=was_training)

def mean_cent_err(parameters: aus.DataParameters, freq_map, output):
    target_frequency = np.array(list(map(aus.map_to_frequency, to_numpy(freq_map))))
    output_frequency = np.array(list(map(aus.map_to_frequency, to_numpy(output))))
    return np.array([abs(aus.cent_diff(target_frequency, output_frequency)) for target_frequency, output_frequency in zip(target_frequency, output_frequency)]).mean()

class ErrorTracker:
    def __init__(self, criterion, eval_funcs, num_validation_batches: int):
        self.criterion = criterion
        self.eval_funcs = eval_funcs
        self.num_validation_batches = num_validation_batches

        self.train_log_losses = []
        self.train_errors = [[] for _ in range(len(eval_funcs))]
        self.train_iter = []

        self.val_log_losses = []
        self.val_errors = [[] for _ in range(len(eval_funcs))]
        self.val_iter = []
    
    def training_update(self, index: int, output, target, label, loss):
        log_loss = np.log10(loss.item())
        self.train_log_losses.append(log_loss)


        output = to_numpy(output)
        target = to_numpy(target)
        for i, eval_func in enumerate(self.eval_funcs):
            func = eval_func["func"]

            self.train_errors[i].append(mean_minibatch_err(output, target, label, func))
        self.train_iter.append(index)

    def validation_update(self, index: int, net: torch.nn.Module, validation_loader: DataLoader):
        val_loss, val_errors = test_net(net, validation_loader, self.criterion, self.num_validation_batches, self.eval_funcs)
        self.val_log_losses.append(np.log10(val_loss))
        for i in range(len(self.val_errors)):
            self.val_errors[i].append(val_errors[i])
        self.val_iter.append(index)
    
    def train_history(self):
        return self.train_iter, self.train_log_losses, self.train_errors
    
    def validation_history(self):
        return self.val_iter, self.val_log_losses, self.val_errors

    def train_history_table(self):
        assert len(self.train_iter) == len(self.train_log_losses)
        train_errors = list(map(list, zip(*self.train_errors)))
        assert len(self.train_iter) == len(train_errors)
        return [[self.train_iter[i], self.train_log_losses[i], *train_errors[i]] for i in range(len(self.train_iter))]
    
    def validation_history_table(self):
        assert len(self.val_iter) == len(self.val_log_losses)
        val_errors = list(map(list, zip(*self.val_errors)))
        assert len(self.val_iter) == len(val_errors)
        return [[self.val_iter[i], self.val_log_losses[i], *val_errors[i]] for i in range(len(self.val_iter))]

def save_model(path: str, file_name: str, net: nn.Module):
    Path(path).mkdir(parents=True, exist_ok=True)
    path = f"{path}/{file_name}"
    model_scripted = torch.jit.script(net)
    model_scripted.save(path)
