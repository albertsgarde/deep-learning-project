import audio_samples_py as aus
import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate
import numpy as np
import itertools

use_cuda = None
device = None

def setup_device(use_cuda_if_possible):
    global use_cuda
    global device
    use_cuda = torch.cuda.is_available() and use_cuda_if_possible
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Running GPU.") if use_cuda else print("No GPU available.")
    return device, use_cuda


def custom_collate(batch):
    signals, ffts, targets, labels = list(zip(*batch))
    return default_collate(list(signals)), default_collate(list(ffts)), default_collate(list(targets)), list(labels)


class AudioDataSet(torch.utils.data.Dataset):
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
    training_loader = torch.utils.data.DataLoader(AudioDataSet(training_parameters, label_to_target), **data_loader_params)
    validation_parameters = parameters.with_seed_offset(seed + 1)
    validation_loader = torch.utils.data.DataLoader(AudioDataSet(validation_parameters, label_to_target), **data_loader_params)

    return training_parameters, training_loader, validation_parameters, validation_loader

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
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        variable = Variable(torch.from_numpy(x))
        if use_cuda:
            variable = variable.cuda()
        return variable
    else:
        raise Exception(f"Unsupported type for to_torch: {type(x)}")
    

def mean_minibatch_err(output, target, error_function):
    assert output.shape == target.shape
    total = 0
    for i, output_row in enumerate(output):
        total += error_function(output_row, target[i,:])
    return total*1./output.shape[0]

def test_net(net, validation_loader, criterion, num_validation_batches, error_functions):
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
    total_errors = [0] * len(error_functions)
    for signal, fft, target, _ in itertools.islice(validation_loader, num_validation_batches):
        signal = signal.to(device)
        fft = fft.to(device)
        target = target.to(device)
        output = net(signal, fft)
        
        total_loss += criterion(output, target)

        output = to_numpy(output)
        target = to_numpy(target)
        for i, error_function in enumerate(error_functions):
            total_errors[i] += mean_minibatch_err(output, target, error_function)

    net.train(mode=was_training)
    return total_loss.item()/num_validation_batches, list(map(lambda x: x / num_validation_batches, total_errors))

def manual_test(net, validation_loader, num_samples, output_functions):
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

def mean_cent_err(parameters, freq_map, output):
    target_frequency = np.array(list(map(parameters.map_to_frequency, to_numpy(freq_map))))
    output_frequency = np.array(list(map(parameters.map_to_frequency, to_numpy(output))))
    return np.array([abs(aus.cent_diff(target_frequency, output_frequency)) for target_frequency, output_frequency in zip(target_frequency, output_frequency)]).mean()