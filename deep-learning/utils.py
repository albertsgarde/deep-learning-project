import audio_samples_py as aus
import torch
import numpy as np

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