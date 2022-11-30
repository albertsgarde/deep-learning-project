import sys
import os
from pathlib import Path
import csv
import json
import itertools
import numpy as np
import torch
import torch.optim as optim

import audio_samples_py as aus

import utils.plots as plots
import utils.criterion as chord_criterion
import utils.utils as utils

def path_from_config(base_dir, path):
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(base_dir, path)

device, use_cuda = utils.setup_device(use_cuda_if_possible = True)
print(f"Using cuda: {use_cuda}")


# Config
config_path = sys.argv[1]
base_dir = os.path.dirname(config_path)

with open(config_path) as config_file:
    config = json.load(config_file)


# Data 
data_config = config["data"]
BATCH_SIZE = data_config["batch_size"]
SEED = data_config["seed"] # Generates different data if changed. Useful to ensure that a result isn't a fluke.

data_parameters_path = path_from_config(base_dir, data_config["parameters_path"])
with open(data_parameters_path) as file:
    json_text = file.read()
    data_parameters = aus.load_data_parameters(json_text)

def label_to_target(label: aus.DataPointLabel):
    target = np.zeros(aus.num_chord_types() + 12, dtype=np.float32)
    target[label.chord_type()] = 1
    target[aus.num_chord_types() + label.note()] = 1
    return target

training_parameters, training_loader, validation_parameters, validation_loader = utils.init_synth_data(data_parameters, label_to_target, SEED, BATCH_SIZE)


# Model
model_path = path_from_config(base_dir, config["model_input_path"])
net = torch.jit.load(model_path)
net.cuda()
print("Successfully loaded model.")


# Error Tracking
def type_accuracy(output, target):
    output_chord_type = np.argmax(output[:aus.num_chord_types()])
    target_chord_type = np.argmax(target[:aus.num_chord_types()])
    return 1 if output_chord_type == target_chord_type else 0

def tone_accuracy(output, target):
    output_chord_tone = np.argmax(output[aus.num_chord_types():])
    target_chord_tone = np.argmax(target[aus.num_chord_types():])
    return 1 if output_chord_tone == target_chord_tone else 0

def total_accuracy(output, target):
    return type_accuracy(output, target) * tone_accuracy(output, target)

eval_funcs = [
    {
        "label": "Type Accuracy",
        "ylim": (0,1),
        "func": lambda output, target, label: type_accuracy(output, target)
    },
    {
        "label": "Tone Accuracy",
        "ylim": (0,1),
        "func": lambda output, target, label: tone_accuracy(output, target)
    },
    {
        "label": "Accuracy",
        "ylim": (0,1),
        "func": lambda output, target, label: total_accuracy(output, target)
    }
]   


# Training
training_config = config["training"]
LEARNING_RATE = training_config["learning_rate"]
WEIGHT_DECAY = training_config["weight_decay"]

criterion = chord_criterion.ChordToneLoss(aus.num_chord_types())  
optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) 

NUM_BATCHES = training_config["num_batches"]
EVAL_EVERY = training_config["eval_every"]
LOG_TRAIN_EVERY = training_config["log_train_every"]
NUM_VALIDATION_BATCHES = training_config["num_validation_batches"]
SAVE_LOGS_EVERY = training_config["save_logs_every"]
SAVE_MODEL_EVERY = training_config["save_model_every"]

TRAIN_LOG_PATH = path_from_config(base_dir, training_config["train_log_path"])
VAL_LOG_PATH = path_from_config(base_dir, training_config["val_log_path"])

MODEL_OUTPUT_DIR = path_from_config(base_dir, config["model_output_dir"])

def save_model(path, net, batch_num):
    utils.save_model(path, f"model_{batch_num}.pt", net)

def save_training_history(train_log_path, val_log_path, error_tracker):
        train_history = error_tracker.train_history_table()
        val_history = error_tracker.validation_history_table()

        with open(train_log_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(train_history)

        with open(val_log_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(val_history)

error_tracker = utils.ErrorTracker(criterion, eval_funcs, NUM_VALIDATION_BATCHES)

print("Starting training...")
net.train()
for i, (signal, fft, target, label) in enumerate(itertools.islice(training_loader, NUM_BATCHES+1)):
    print(f"Batch num: {i}/{NUM_BATCHES}", end="\r")
    if i%EVAL_EVERY == 0:
        error_tracker.validation_update(i, net, validation_loader)

    signal = utils.to_torch(signal)
    fft = utils.to_torch(fft)
    target = utils.to_torch(target)
    output = net(signal, fft)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if i%LOG_TRAIN_EVERY == 0:
        error_tracker.training_update(i, output, target, label, loss)
    
    if i % SAVE_MODEL_EVERY == 0:
        save_model(MODEL_OUTPUT_DIR, net, i)

    if i % SAVE_LOGS_EVERY == 0:
        save_training_history(TRAIN_LOG_PATH, VAL_LOG_PATH, error_tracker)

save_model(MODEL_OUTPUT_DIR, net, NUM_BATCHES)
save_training_history(TRAIN_LOG_PATH, VAL_LOG_PATH, error_tracker)
print(f"Finished training on {NUM_BATCHES} batches and saved model to '{MODEL_OUTPUT_DIR}'.")

val_loss, [val_type_accuracy, val_tone_accuracy, val_accuracy] = utils.test_net(net, validation_loader, criterion, NUM_VALIDATION_BATCHES, eval_funcs)
print(f"Loss={val_loss}, Type Accuracy={val_type_accuracy:.3f}, Tone Accuracy={val_tone_accuracy:.3f}, Accuracy={val_accuracy:.3f}")
