from .audio_samples_py import *
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import os

def plot_data_point(data_point):
    plt.figure(figsize=(10, 4))
    plt.plot(data_point.get_samples())
    plt.ylim(-1, 1)
    plt.show()

def play_data_point(data_point):
    data_point.audio_to_file("play.wav")
    display(Audio("play.wav", autoplay=True))
    os.remove("play.wav")