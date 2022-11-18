import os
import torch

import audio_samples_py as aus

import utils.plots as plots
import utils.criterion as chord_criterion
import utils.utils as utils

device, use_cuda = utils.setup_device(use_cuda_if_possible = True)

# Data 
SAMPLE_LENGTH = 1024
BATCH_SIZE = 64
SEED = 1 # Generates different data if changed. Useful to ensure that a result isn't a fluke.

possible_chord_types = [i for i in range(aus.num_chord_types())]
octave_parameters = aus.OctaveParameters(add_root_octave_probability=0.5,
        add_other_octave_probability=0.3)
parameters = aus.DataParameters(num_samples=SAMPLE_LENGTH, octave_parameters=octave_parameters, min_frequency=50, max_frequency=2000, min_frequency_std_dev=0.5, max_frequency_std_dev=3., possible_chord_types=possible_chord_types) \
    .add_sine(probability=0.5, amplitude_range=(0.1,0.2)) \
    .add_saw(probability=0.5, amplitude_range=(0.1, 0.2)) \
    .add_pulse(probability=0.5, amplitude_range=(0.1, 0.2), duty_cycle_range=(0.1, 0.9)) \
    .add_triangle(probability=0.5, amplitude_range=(0.1, 0.2)) \
    .add_noise(probability=1, amplitude_range=(0.001, 0.04)) \
    .apply_distortion(probability=0.5, power_range=(0.1, 20)) \
    .apply_normalization(probability=1)

def label_to_target(label: aus.DataPointLabel):
    target = np.zeros(aus.num_chord_types() + 12, dtype=np.float32)
    target[label.chord_type()] = 1
    target[aus.num_chord_types() + label.note()] = 1
    return target

training_parameters, training_loader, validation_parameters, validation_loader = utils.init_synth_data(parameters, label_to_target, SEED, BATCH_SIZE)

# Model
model_path = os.environ['MODEL_PATH']
net = torch.jit.load(model_path)
net.cuda()
