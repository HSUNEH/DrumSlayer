import torch 
import torch.nn as nn
import numpy as np
import pretty_midi

# # read 0.npy file
# data = np.load('/workspace/DrumSlayer/DrumTranscriber/0.npy')
# breakpoint()
# print(data.shape)

import os
import scipy
import dac
from dac.utils import load_model
from dac.model import DAC
import scipy.io.wavfile

file = '/disk2/st_drums/one_shots/test/kick/ukg_kick_mustard_codes.npy'
data = np.load(file) #(2, 9, 26)
inst_tensor = torch.tensor(data)
    
dac_model_path = dac.utils.download(model_type="44khz")
dac_model = dac.DAC.load(dac_model_path) 
dac_model.eval()


for i, codes in enumerate([inst_tensor]):
    breakpoint()
    inst_codes = codes[0].unsqueeze(0).long() # .permute(0,2,1) # torch.Size([1, 9, seq_len])
    latent = dac_model.quantizer.from_codes(inst_codes)[0]
    audio = dac_model.decode(latent)[0]
    audio = audio.detach().numpy().astype(np.float32)

    output_dir = f'/workspace/DrumSlayer/DrumTranscriber/test/check.wav'

    os.makedirs(os.path.dirname(output_dir), exist_ok=True) #(1, 18432)
    scipy.io.wavfile.write(output_dir, 44100, audio.T)
    breakpoint()