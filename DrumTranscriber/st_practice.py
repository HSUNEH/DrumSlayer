import torch 
import torch.nn as nn
import numpy as np
import pretty_midi

# # read 0.npy file
# data = np.load('/workspace/DrumSlayer/DrumTranscriber/0.npy')
# breakpoint()
# print(data.shape)

a = torch.ones(4,2,220500)
a = a[:,:,:176400]
print(12//2)

