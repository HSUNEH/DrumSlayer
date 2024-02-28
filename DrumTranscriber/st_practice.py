import torch 
import torch.nn as nn
import numpy as np
import pretty_midi
def tokenize_midi(kick_midi, snare_midi, hihat_midi):
    midi_tokens = []
    for type, midi in enumerate([kick_midi, snare_midi, hihat_midi]):
        for note in midi.instruments[0].notes:
            onset = int(note.start // 0.005) # Time resolution is 5ms
            vel = int(note.velocity)
            midi_tokens.append([4+onset, 4+1000+vel, 4+1000+128+type]) # 2 is reserved for start and end tokens.
            
    midi_tokens.sort(key=lambda x: x[0]) # Sort by onset time.
    midi_tokens = [item for sublist in midi_tokens for item in sublist] # Flatten.
    # midi_tokens_np = np.ones(152, dtype=np.int32) * 2 # <PAD> token # max_len => 152
    # midi_tokens_np[1:len(midi_tokens)+1] = np.array(midi_tokens, dtype=np.int32)
    # midi_tokens_np[0] = 0 # SOS token 
    # midi_tokens_np[len(midi_tokens)+1] = 1 # EOS token
    return len(midi_tokens) #midi_tokens_np


file_path = '/disk2/st_drums/generated_data/drum_data_train/generated_midi/'
min_len = 90 
from tqdm import tqdm

for i in tqdm(range(900000)):   
    kick_midi =  pretty_midi.PrettyMIDI(file_path + f"kick_midi/kick_midi_{i}.midi")
    snare_midi =  pretty_midi.PrettyMIDI(file_path + f"snare_midi/snare_midi_{i}.midi")
    hihat_midi = pretty_midi.PrettyMIDI(file_path + f"hihat_midi/hihat_midi_{i}.midi")
    midi_len = tokenize_midi(kick_midi, snare_midi, hihat_midi)
    min_len = min(min_len, midi_len)
    
print(min_len)


