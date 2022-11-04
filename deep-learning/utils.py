import audio_samples_py as aus
import torch
from torch.autograd import Variable
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

class AudioDataSet(torch.utils.data.Dataset):
    def __init__(self, parameters: aus.DataParameters):
         self.parameters = parameters

    def __len__(self):
        return np.iinfo(np.int64).max
    
    def __getitem__(self, index):
        data_point = self.parameters.generate_at_index(index)
        return data_point.samples(), data_point.audio().fft(), data_point.label()

def init_synth_data(parameters: aus.DataParameters, seed: int, batch_size: int):
    assert seed >= 0, f"seed must be non-negative. seed={seed}"
    assert batch_size > 0, f"batch_size must be positive. batch_size={batch_size}"

    data_loader_params = {"batch_size": batch_size}

    training_parameters = parameters.with_seed_offset(seed)
    training_loader = torch.utils.data.DataLoader(AudioDataSet(training_parameters), **data_loader_params)
    validation_parameters = parameters.with_seed_offset(seed + 1)
    validation_loader = torch.utils.data.DataLoader(AudioDataSet(validation_parameters), **data_loader_params)

    return training_parameters, training_loader, validation_parameters, validation_loader

def to_numpy(x):
    """ Get numpy array for both cuda and not. """
    global use_cuda
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()

def to_torch(x):
    variable = Variable(torch.from_numpy(x))
    if use_cuda:
        variable = variable.cuda()
    return variable

def mean_cent_err(parameters, freq_map, output):
    target_frequency = np.array(list(map(parameters.map_to_frequency, to_numpy(freq_map))))
    output_frequency = np.array(list(map(parameters.map_to_frequency, to_numpy(output))))
    return np.array([abs(aus.cent_diff(target_frequency, output_frequency)) for target_frequency, output_frequency in zip(target_frequency, output_frequency)]).mean()

def test_net(net, parameters, validation_loader, criterion, num_validation_batches):
    global device
    was_training = net.training
    net.eval()
    total_loss = 0
    total_cent_diff = 0
    for signal, fft, freq_map in itertools.islice(validation_loader, num_validation_batches):
        signal = signal.to(device)
        fft = fft.to(device)
        freq_map = freq_map.to(device)
        output = net(signal, fft)
        
        total_loss += criterion(output, freq_map)
        total_cent_diff += mean_cent_err(parameters, freq_map, output)
    net.train(mode=was_training)
    return total_loss.item()/num_validation_batches, total_cent_diff/num_validation_batches

def manual_test(net, validation_parameters, num_iterations):
    was_training = net.training
    net.eval()
    for data_point in [validation_parameters.generate_at_index(i) for i in range(num_iterations) ]:
        samples = to_torch(data_point.samples()).unsqueeze(0)
        fft = to_torch(data_point.audio().fft()).unsqueeze(0)
        freq_map = to_torch(np.array([data_point.frequency_map()])).unsqueeze(0)
        output = net(samples, fft)
        target_frequency = validation_parameters.map_to_frequency(freq_map.item())
        output_frequency = validation_parameters.map_to_frequency(output.item())
        print("Frequency: {:.2f} Output: {:.2f} Cent diff: {:.2f}".format(target_frequency, output_frequency, aus.cent_diff(target_frequency, output_frequency)))
    net.train(mode=was_training)