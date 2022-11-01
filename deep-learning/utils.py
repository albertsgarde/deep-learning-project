import audio_samples_py as aus
import torch
import numpy as np

def setup_device(use_cuda_if_possible):
    use_cuda = torch.cuda.is_available() and use_cuda_if_possible
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Running GPU.") if use_cuda else print("No GPU available.")
    return device

class AudioDataSet(torch.utils.data.Dataset):
    def __init__(self, parameters: aus.DataParameters):
         self.parameters = parameters

    def __len__(self):
        return np.iinfo(np.int64).max
    
    def __getitem__(self, index):
        data_point = self.parameters.generate_at_index(index)
        return data_point.samples(), data_point.audio().fft(), torch.tensor([data_point.frequency_map()]).unsqueeze(0)

def init_synth_data(parameters: aus.DataParameters, seed: int, batch_size: int):
    assert seed >= 0, f"seed must be non-negative. seed={seed}"
    assert batch_size > 0, f"batch_size must be positive. batch_size={batch_size}"

    data_loader_params = {"batch_size": batch_size}

    training_parameters = parameters.with_seed_offset(seed)
    training_loader = torch.utils.data.DataLoader(AudioDataSet(training_parameters), **data_loader_params)
    validation_parameters = parameters.with_seed_offset(seed + 1)
    validation_loader = torch.utils.data.DataLoader(AudioDataSet(validation_parameters), **data_loader_params)

    return training_parameters, training_loader, validation_parameters, validation_loader

def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()