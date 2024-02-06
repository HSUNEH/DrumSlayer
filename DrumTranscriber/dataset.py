from torch.utils.data import Dataset
import glob
import numpy
import pretty_midi
import os
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F
import numpy as np
import pretty_midi
from einops import rearrange

class DrumSlayerDataset(Dataset):
    def __init__(self, file_path, split, audio_encoding_type, max_len=152):
        assert audio_encoding_type in ["latents", "codes", "z"] # dim: 72, 9, 1024
        self.file_path = file_path
        self.split = split
        self.max_len = max_len
        self.encoding_type = audio_encoding_type
        self.num_data = len(glob.glob(file_path + f"drum_data_{split}/mixed_loops/*.wav"))
    
    def __getitem__(self, idx):
        audio_rep = np.load(self.file_path + f"drum_data_{self.split}/mixed_loops/{idx}_{self.encoding_type}.npy") ## npy 생성 -> preprocess_dac.py
        audio_rep = rearrange(audio_rep, 'c d t -> t c d') # c: channel, d: dim, t: time
        kick_midi =  pretty_midi.PrettyMIDI(self.file_path + f"drum_data_{self.split}/generated_midi/kick_midi/kick_midi_{idx}.midi")
        snare_midi =  pretty_midi.PrettyMIDI(self.file_path + f"drum_data_{self.split}/generated_midi/snare_midi/snare_midi_{idx}.midi")
        hihat_midi = pretty_midi.PrettyMIDI(self.file_path + f"drum_data_{self.split}/generated_midi/hihat_midi/hihat_midi_{idx}.midi")
        midi_tokens = self.tokenize_midi(kick_midi, snare_midi, hihat_midi)
        return audio_rep, midi_tokens

    def __len__(self):

        return self.num_data

    def tokenize_midi(self, kick_midi, snare_midi, hihat_midi):
        midi_tokens = []
        for type, midi in enumerate([kick_midi, snare_midi, hihat_midi]):
            for note in midi.instruments[0].notes:
                onset = int(note.start // 0.005) # Time resolution is 5ms
                vel = int(note.velocity)
                midi_tokens.append([3+onset, 3+1000+vel, 3+1000+128+type]) # 2 is reserved for start and end tokens.
        midi_tokens.sort(key=lambda x: x[0]) # Sort by onset time.
        midi_tokens = [item for sublist in midi_tokens for item in sublist] # Flatten.
        midi_tokens_np = np.ones(self.max_len, dtype=np.int32) * 2 # <PAD> token
        midi_tokens_np[1:len(midi_tokens)+1] = np.array(midi_tokens, dtype=np.int32)
        midi_tokens_np[0] = 0 # SOS token 
        midi_tokens_np[len(midi_tokens)+1] = 1 # EOS token
        return midi_tokens_np
    
if __name__ == "__main__":
    data_dir = '/workspace/DrumSlayer/generated_data/'
    dataset = DrumSlayerDataset(data_dir, "train", "codes")
    x = dataset[0]
    pass
