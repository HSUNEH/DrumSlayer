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
        audio_rep = np.load(self.file_path + f"drum_data_{self.split}/mixed_loops/{idx}_{self.encoding_type}.npy") ## dac npy 생성 -> preprocess_dac.py
        audio_rep = rearrange(audio_rep, 'c d t -> t c d') # c: channel, d: dim, t: time
        kick_midi =  pretty_midi.PrettyMIDI(self.file_path + f"drum_data_{self.split}/generated_midi/kick_midi/kick_midi_{idx}.midi")
        snare_midi =  pretty_midi.PrettyMIDI(self.file_path + f"drum_data_{self.split}/generated_midi/snare_midi/snare_midi_{idx}.midi")
        hihat_midi = pretty_midi.PrettyMIDI(self.file_path + f"drum_data_{self.split}/generated_midi/hihat_midi/hihat_midi_{idx}.midi")
        
        # TODO : list 읽고 해당 Oneshot 의 dac 파일 불러와서 적용 
        # TODO : preprocess 에서 one shot dac 파일 생성
        def load_dac(file_path, idx):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                oneshot_name = lines[idx].strip()  # Get the content at index idx and remove leading/trailing whitespaces
            return oneshot_name

        # Usage
        oneshot_path = '/disk2/st_drums/one_shots/'
        kick_name = load_dac(self.file_path + f"drum_data_{self.split}/kickShotList.txt", idx)
        kick_dac = np.load(oneshot_path + f"{self.split}/kick/{kick_name}_{self.encoding_type}.npy") #snare, hhclosed # (2,9,431)
        kick_dac_l = kick_dac[0] # (9, 431)
        kick_dac_r = kick_dac[1] # (9, 431)
        
        snare_name = load_dac(self.file_path + f"drum_data_{self.split}/snareShotList.txt", idx)
        snare_dac = np.load(oneshot_path + f"{self.split}/snare/{snare_name}_{self.encoding_type}.npy") 
        snare_dac_l = snare_dac[0] # (9, 431)
        snare_dac_r = snare_dac[1] # (9, 431)
        
        hihat_name = load_dac(self.file_path + f"drum_data_{self.split}/hihatShotList.txt", idx)
        hihat_dac = np.load(oneshot_path + f"{self.split}/hhclosed/{hihat_name}_{self.encoding_type}.npy")
        hihat_dac_l = hihat_dac[0] # (9, 431)
        hihat_dac_r = hihat_dac[1] # (9, 431)
        
        audio_tokens = self.tokenize_audio_stereo(kick_midi, snare_midi, hihat_midi, kick_dac_l, kick_dac_r, snare_dac_l, snare_dac_r, hihat_dac_l, hihat_dac_r)
        
        
        # midi_tokens = self.tokenize_midi(kick_midi, snare_midi, hihat_midi)
        # return audio_rep, midi_tokens
        return audio_rep, audio_tokens 

    def __len__(self):

        return self.num_data

    def tokenize_midi(self, kick_midi, snare_midi, hihat_midi):
        midi_tokens = []
        for type, midi in enumerate([kick_midi, snare_midi, hihat_midi]):
            for note in midi.instruments[0].notes:
                onset = int(note.start // 0.005) # Time resolution is 5ms
                vel = int(note.velocity)
                midi_tokens.append([4+onset, 4+1000+vel, 4+1000+128+type]) # 2 is reserved for start and end tokens.
        midi_tokens.sort(key=lambda x: x[0]) # Sort by onset time.
        midi_tokens = [item for sublist in midi_tokens for item in sublist] # Flatten.
        midi_tokens_np = np.ones(self.max_len, dtype=np.int32) * 2 # <PAD> token
        midi_tokens_np[1:len(midi_tokens)+1] = np.array(midi_tokens, dtype=np.int32)
        midi_tokens_np[0] = 0 # SOS token 
        midi_tokens_np[len(midi_tokens)+1] = 1 # EOS token
        return midi_tokens_np

    def tokenize_audio_stereo(kick_midi, snare_midi, hihat_midi, kick_dac_l, kick_dac_r, snare_dac_l, snare_dac_r, hihat_dac_l, hihat_dac_r):
        midi_vocab_size = 1000+128+4+1 # 1133
        audio_vocal_size = 1024 + 1 # 1025

        midi_tokens = []
        for type, midi in enumerate([kick_midi, snare_midi, hihat_midi]):
            for note in midi.instruments[0].notes:
                onset = int(note.start // 0.005) # Time resolution is 5ms
                vel = int(note.velocity)
                midi_tokens.append([4+onset, 4+1000+vel, 4+1000+128+type]) # 2 is reserved for start and end tokens.
        midi_tokens.sort(key=lambda x: x[0]) # Sort by onset time.
        midi_tokens = [item for sublist in midi_tokens for item in sublist] # Flatten.
        all_tokens_np = np.ones((1+(kick_dac_l.shape[0])*2,152+(431+8+1)*3), dtype=np.int32) * 2   # <PAD> token = 2 
        
        all_tokens_np[0,1:len(midi_tokens)+1] = np.array(midi_tokens, dtype=np.int32) 
        all_tokens_np[:,0] = 0 # <SOS> token
        
        # TODO : interleaving pattern
        for type, dac_list  in enumerate([[kick_dac_l, kick_dac_r], [snare_dac_l, snare_dac_r], [hihat_dac_l, hihat_dac_r]]):# with latents
            for lr, dac in enumerate(dac_list):
                for i, codes in enumerate(dac): # dac : (9, 431)
                    start = i+153+type*(431+8+1)
                    end = start + dac.shape[1] 
                    if lr == 0:
                        all_tokens_np[1+i, start: end] = codes+2000    
                    else:
                        all_tokens_np[10+i, start: end] = codes+2000    
                    
        for i in range(3):
            all_tokens_np[:,152+(431+8+1)*i] = 3 # <SEP> token 

        all_tokens_np[:,-1] = 1 # <EOS> token

        return all_tokens_np # (19, 1472) # sep, eos at 152, 592, 1032, 1472

if __name__ == "__main__":
    data_dir = '/workspace/DrumSlayer/generated_data/'
    dataset = DrumSlayerDataset(data_dir, "train", "codes")
    x = dataset[0]
    pass
