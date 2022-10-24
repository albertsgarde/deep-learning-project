from .audio_samples_py import *
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import os

def plot_audio(audio):
    plt.figure(figsize=(10, 4))
    plt.plot(audio.samples())
    plt.ylim(-1, 1)
    plt.show()

def plot_fft(audio):
    plt.figure(figsize=(10, 4))
    plt.plot(audio.fft())
    plt.ylim(-1, 1)
    plt.show()

def plot_data_point(data_point):
    plot_audio(data_point.audio())

def plot_data_point_fft(data_point):
    plot_fft(data_point.audio())

def play_audio(audio):
    audio.to_wav("play.wav")
    display(Audio("play.wav", autoplay=True))
    os.remove("play.wav")


def play_data_point(data_point):
    play_audio(data_point.audio())