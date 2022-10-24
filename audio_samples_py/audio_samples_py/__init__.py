from .audio_samples_py import *
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import os

def load_audio(path):
    return Audio.from_wav(path)

def plot_audio(audio):
    plt.figure(figsize=(10, 4))
    plt.plot(audio.samples())
    plt.ylim(-1, 1)
    plt.show()

def plot_data_point(data_point):
    plot_audio(data_point.audio())

def play_audio(audio):
    audio.to_wav("play.wav")
    display(Audio("play.wav", autoplay=True))
    os.remove("play.wav")


def play_data_point(data_point):
    play_audio(data_point.audio())