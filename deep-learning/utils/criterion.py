import torch
from torch import nn

class ChordToneLoss(nn.Module):
    def __init__(self, num_chord_types: int, chord_type_loss: nn.Module = nn.CrossEntropyLoss(), chord_tone_loss: nn.Module = nn.CrossEntropyLoss(), chord_type_coef: float = 0.5):
        super(ChordToneLoss, self).__init__()
        assert num_chord_types > 0, f"Number of chord types must be at least 1. Number of chord types: {num_chord_types}"
        assert chord_type_coef >= 0, f"Coefficient for chord type loss must be at least 0. Chord type coef: {chord_type_coef}"
        assert chord_type_coef <= 1, f"Coefficitnet for chord type loss must be no more than 1. Chord type coef: {chord_type_coef}"
        self.num_chord_types = num_chord_types
        self.chord_type_loss = chord_type_loss
        self.chord_tone_loss = chord_tone_loss

        self.chord_type_coef = chord_type_coef
        self.chord_tone_coef = 1-chord_type_coef


    def forward(self, output, target):
        assert output.shape[1] == (self.num_chord_types + 12), f"Length of outputs must be equal to number of chord types plus number of notes. Length of outputs: {output.shape[1]}  Number of chord types: {self.num_chord_types}"
        assert target.shape[1] == (self.num_chord_types + 12), f"Length of targets must be equal to number of chord types plus number of notes. Length of targets: {target.shape[1]}  Number of chord types: {self.num_chord_types}"

        output_chord_type = output[:,:self.num_chord_types]
        target_chord_type = target[:,:self.num_chord_types]

        output_chord_tone = output[:,self.num_chord_types:]
        target_chord_tone = target[:,self.num_chord_types:]

        chord_type_loss = self.chord_type_loss(output_chord_type, target_chord_type)
        chord_tone_loss = self.chord_tone_loss(output_chord_tone, target_chord_tone)

        total_loss = self.chord_type_coef*chord_type_loss + self.chord_tone_coef*chord_tone_loss

        return total_loss

