import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

import numpy as np
import librosa
import os

class SpectrogramDataset(Dataset):
    def __init__(self, data_folder, sr=48000, n_mels=128):
        self.data_folder = data_folder
        self.sr = sr
        self.n_mels = n_mels
        
        self.wav_folder = os.path.join(data_folder,'samples')
        self.wav_files = [file for file in os.listdir(self.wav_folder) if file.endswith('.wav')]
        
        self.midi_nps_folder = os.path.join(data_folder,'generated_midi_numpy')
        self.kick_nps_folder = os.path.join(self.midi_nps_folder, 'kick_16')
        self.kick_nps = [file for file in os.listdir(self.kick_nps_folder) if file.endswith('.npy')]
        self.hihat_nps_folder = os.path.join(self.midi_nps_folder, 'hihat_16')
        self.hihat_nps = [file for file in os.listdir(self.hihat_nps_folder) if file.endswith('.npy')]
        self.snare_nps_folder = os.path.join(self.midi_nps_folder, 'snare_16')
        self.snare_nps = [file for file in os.listdir(self.snare_nps_folder) if file.endswith('.npy')]

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_file = os.path.join(self.wav_folder, self.wav_files[idx])
        
        # spectrogram = audio_to_log_mel_spectrogram(wav_file, sr=self.sr, n_mels=self.n_mels)
        spectrogram, sr = librosa.load(wav_file, sr = self.sr) # TODO : change librosa -> audio read
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        # padding for 2^
        # print("spectrogram : ",spectrogram.shape)
        # if spectrogram.shape[0] % 1024 != 0:
        #     padding = spectrogram.shape[0] % 1024
        #     spectrogram = torch.cat([spectrogram, torch.zeros(padding)])
        # print("spectrogram : ",spectrogram.shape)


        kick_np = np.load(os.path.join(self.kick_nps_folder, self.kick_nps[idx]))
        kick_torch = torch.from_numpy(kick_np[0]).unsqueeze(0)
        hihat_np = np.load(os.path.join(self.hihat_nps_folder, self.hihat_nps[idx]))
        hihat_torch = torch.from_numpy(hihat_np[0]).unsqueeze(0)
        snare_np = np.load(os.path.join(self.snare_nps_folder, self.snare_nps[idx]))
        snare_torch = torch.from_numpy(snare_np[0]).unsqueeze(0)

        drum_torch = torch.cat([kick_torch, hihat_torch, snare_torch], dim=0)
        return  spectrogram, drum_torch
        # spectrogram : (batch, sound+padding)
        # drum_torch : (batch , 2, 384000)

class DrumTorchDataset(L.LightningDataModule):
    def __init__(self, data_folder, batch_size):
        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
    
    # def prepare_data(self):
    #     pass
  
    def setup(self, stage=None):
        pass
    
    def train_dataloader(self):
        self.df = os.path.join(self.data_folder,'drum_data_train')
        self.dataset = SpectrogramDataset(self.df)
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers = 16)

    def val_dataloader(self):
        self.df = os.path.join(self.data_folder,'drum_data_val')
        self.dataset = SpectrogramDataset(self.df)
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers = 16)
  
    def test_dataloader(self):
        self.df = os.path.join(self.data_folder,'drum_data_test')
        self.dataset = SpectrogramDataset(self.df)
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers = 16)
