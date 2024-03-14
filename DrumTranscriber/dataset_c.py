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
    def __init__(self, file_path, split, audio_encoding_type, args, max_len=152):
        assert audio_encoding_type in ["latents", "codes", "z"] # dim: 72, 9, 1024
        self.file_path = file_path
        self.split = split
        self.max_len = max_len
        self.encoding_type = audio_encoding_type
        self.num_data = len(glob.glob(file_path + f"drum_data_{split}/mixed_loops/*.wav"))
        self.train_type = args.train_type
        
    def __getitem__(self, idx):
        audio_rep = np.load(self.file_path + f"drum_data_{self.split}/mixed_loops/{idx}_{self.encoding_type}.npy") ## dac npy 생성 -> preprocess_dac.py
        audio_rep = rearrange(audio_rep, 'c d t -> t c d') # c: channel, d: dim, t: time

        if self.train_type == "kshm":
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
        kick_name = kick_name.replace('.wav', f'_{self.encoding_type}.npy')
        kick_dac = np.load(oneshot_path + f"{self.split}/kick/{kick_name}") #snare, hhclosed # (2,9,431)
        kick_dac_l = kick_dac[0] # (9, 431)
        # kick_dac_r = kick_dac[1] # (9, 431)
        
        snare_name = load_dac(self.file_path + f"drum_data_{self.split}/snareShotList.txt", idx)
        snare_name = snare_name.replace('.wav', f'_{self.encoding_type}.npy')
        snare_dac = np.load(oneshot_path + f"{self.split}/snare/{snare_name}") 
        snare_dac_l = snare_dac[0] # (9, 431)
        # snare_dac_r = snare_dac[1] # (9, 431)

        hihat_name = load_dac(self.file_path + f"drum_data_{self.split}/hihatShotList.txt", idx)
        hihat_name = hihat_name.replace('.wav', f'_{self.encoding_type}.npy')
        hihat_dac = np.load(oneshot_path + f"{self.split}/hhclosed/{hihat_name}")
        hihat_dac_l = hihat_dac[0] # (9, 431)
        # hihat_dac_r = hihat_dac[1] # (9, 431)

        if self.train_type == "kshm":
            audio_tokens = self.tokenize_audio_mono(kick_midi, snare_midi, hihat_midi, kick_dac_l, snare_dac_l, hihat_dac_l)
        
        elif self.train_type == "ksh":
            audio_tokens, dac_length = self.tokenize_audio(kick_dac_l, snare_dac_l, hihat_dac_l)
        
        elif self.train_type == "kick":
            audio_tokens, dac_length = self.tokenize_inst(kick_dac_l)
        elif self.train_type == "snare":
            audio_tokens, dac_length = self.tokenize_inst(snare_dac_l)
        elif self.train_type == "hihat":
            audio_tokens, dac_length = self.tokenize_inst(hihat_dac_l)
        
        return audio_rep, audio_tokens, dac_length

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

    def tokenize_audio_mono(self, kick_midi, snare_midi, hihat_midi, kick_dac_l, snare_dac_l, hihat_dac_l):
        midi_vocab_size = 1000+128+4+1 # 1133
        audio_vocal_size = 1024+4+1 # 1029

        midi_tokens = []
        for type, midi in enumerate([kick_midi, snare_midi, hihat_midi]):
            for note in midi.instruments[0].notes:
                onset = int(note.start // 0.005) # Time resolution is 5ms
                vel = int(note.velocity)
                if onset <800: # 4sec slice 
                    midi_tokens.append([4+onset, 4+1000+vel, 4+1000+128+type]) # 2 is reserved for start and end tokens.
        midi_tokens.sort(key=lambda x: x[0]) # Sort by onset time.
        midi_tokens = [item for sublist in midi_tokens for item in sublist] # Flatten.
        all_tokens_np = np.ones((1+(kick_dac_l.shape[0]),92+1+(345+8+1)*3), dtype=np.int32) * 2   # <PAD> token = 2 
        
        all_tokens_np[0,1:len(midi_tokens)+1] = np.array(midi_tokens, dtype=np.int32)
        all_tokens_np[:,0] = 0 # <SOS> token
        
        # interleaving pattern
        for type, dac  in enumerate([kick_dac_l, snare_dac_l, hihat_dac_l]):# with latents
            for i, codes in enumerate(dac): # dac : (9, (max)431)
                start = i+93+type*(345+8+1)
                end = start + dac.shape[1] 
                all_tokens_np[1+i, start: end] = codes + 4 # +2000
        
        for i in range(3):
            all_tokens_np[:,92+(345+8+1)*i] = 3 # <SEP> token 

        all_tokens_np[:,-1] = 1 # <EOS> token

        return all_tokens_np # (10, 1155) # sep, eos at 152, 592, 1032, 1472

    def tokenize_audio(self, kick_dac_l, snare_dac_l, hihat_dac_l):
        midi_vocab_size = 1000+128+4+1 # 1133
        audio_vocab_size = 1024+4+1 # 1029

        all_tokens_np = np.ones(((kick_dac_l.shape[0]),1+(345+8+1)*3), dtype=np.int32) * 2   # <PAD> token = 2 
        
        # all_tokens_np[0,1:len(midi_tokens)+1] = np.array(midi_tokens, dtype=np.int32)
        all_tokens_np[:,0] = 0 # <SOS> token
        dac_length = []
        
        # interleaving pattern
        for type, dac  in enumerate([kick_dac_l,snare_dac_l,hihat_dac_l]):# with latents
            inst_length = dac.shape[1]
            for i, codes in enumerate(dac): # dac : (9, (max)431)
                start = 1+i+sum(dac_length)+9*type
                end = start + inst_length
                all_tokens_np[i, start: end] = codes + 4 # +2000      
            dac_length.append(inst_length)
            last_end = end
            for i in range(9):
                all_tokens_np[i, last_end] = 3 # <SEP> token 
                if type == 2:
                    all_tokens_np[i, last_end] = 1# <EOS> token
        # for i in range(1,3):
        #     all_tokens_np[:,(345+8+1)*i] = 3 # <SEP> token 


        return all_tokens_np, dac_length # (10, 1472) # sep, eos at 152, 592, 1032, 1472


    def tokenize_inst(self,inst_dac_l):
        midi_vocab_size = 1000+128+4+1 # 1133
        audio_vocab_size = 1024+4+1 # 1029

        all_tokens_np = np.ones(((inst_dac_l.shape[0]),1+(345+8+1)*1), dtype=np.int32) * 2   # <PAD> token = 2 
        
        # all_tokens_np[0,1:len(midi_tokens)+1] = np.array(midi_tokens, dtype=np.int32)
        all_tokens_np[:,0] = 0 # <SOS> token
        dac_length = inst_dac_l.shape[1]
        # interleaving pattern
        for type, dac  in enumerate([inst_dac_l]):# with latents
            for i, codes in enumerate(dac): # dac : (9, (max)431)
                start = i+1+type*(345+8+1)
                end = start + dac.shape[1]
                all_tokens_np[i, start: end] = codes + 4 # +2000
        
        # for i in range(3):
        #     all_tokens_np[:,(345+8+1)*i] = 3 # <SEP> token 

        all_tokens_np[:,dac_length+9] = 1 # <EOS> token

        return all_tokens_np, dac_length # (10, 1472) # sep, eos at 152, 592, 1032, 1472

if __name__ == "__main__":
    data_dir = '/workspace/DrumSlayer/generated_data/'
    dataset = DrumSlayerDataset(data_dir, "train", "codes")
    x = dataset[0]
    pass