# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 11:58:17 2022

@author: Ejer
"""
#44100 sample rate

import os
import json
import pydub
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.io.wavfile import write
import numpy as np

def mapNote(folder):
    note=folder.replace("m", "")
    return {
        'a': 0,
        'b': 2,
        'c': 3,
        'd': 5,
        'e': 7,
        'f': 8,
        'g': 10
         }[note]

path = "C:\\Users\\Ejer\\Documents\\Deep learning\\Guitar_Only"
newpath = "C:\\Users\\Ejer\\Documents\\Deep learning\\projekt\\deep-learning-project\\deep-learning\\data\\short_guitar_samples"
dir = os.listdir(path)

num_samples = 512
sample_rate = 44100

tuple_list = []

mol = False

for folder in dir:
    note = mapNote(folder)
    mol = "m" in folder
    type = 2 if mol else 1
    path2 = path + "\\" +folder
    files = os.listdir(path2)
    for file in files:
        path3 = path2+"\\"+file
        name = file.replace(".wav", "")
        sample_dict = {"sample_rate": sample_rate,
               "base_frequency": None,
               "note": note,
               "chord_type": type,
               "num_samples": num_samples
               }
        sample_tuple = (name+"_short", sample_dict)
        tuple_list.append(sample_tuple)
        
        #cut data from start
 
        sound = AudioSegment.from_file(path3)
        cut = num_samples/sample_rate*1000
        start_point = len(sound) / 4.0
        sound_short = sound[start_point:start_point+cut]
        sound_short_a = sound_short.get_array_of_samples()
        sound_short_a = np.array(sound_short_a, dtype=np.float32)
        write(newpath + "\\" + name +"_short.wav",sample_rate,sound_short_a)
        #sound_short.export(out_f = ( newpath + "\\" + name +"_short.wav"), format = "wav")

json_data = json.dumps(tuple_list,indent=4)

with open(newpath +"\\_labels.json", "w") as outfile:
    outfile.write(json_data)

print(json_data)



