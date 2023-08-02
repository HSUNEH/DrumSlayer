# TODO 1) audio -> Log mel spectrogram -> encoding -> decoding -> MIDI file
# Baseline 1) : Encoder Decoder (Onset + DrummerNet)
## Loss
# Onset : 
# DrummerNet : Mean Absolute Error between audio CQT

# TODO 2): MIDI file  <-> midi token 
# Baseline 2) : MT3 - T5 transformer 


# audiopreprocessing :  audio -> log mel
# model :               encoding & decoding

## pytorch lightning 
# essential 
# 1) Model (with train_step, configure_optimizers)
# 2) dataset
# 3) trainer = L.Trainer()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import lightning as L
from audiopreprocessing import audio_to_log_mel_spectrogram
import librosa
import numpy
import os


class Autoencoder(L.LightningModule):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128*188, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Bottleneck layer
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128*188)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        y_pred = self(x)
        loss = nn.MSELoss()(y_pred, x)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)



class SpectrogramDataset(Dataset):
    def __init__(self, data_folder, sr=48000, n_mels=128):
        self.data_folder = data_folder
        self.sr = sr
        self.n_mels = n_mels
        self.wav_files = [file for file in os.listdir(data_folder) if file.endswith('.wav')]

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_file = os.path.join(self.data_folder, self.wav_files[idx])
        spectrogram = audio_to_log_mel_spectrogram(wav_file, sr=self.sr, n_mels=self.n_mels)
        return torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)



if __name__ == "__main__":
    # Parameters
    n_mels=128
    hop_length=512 
    n_fft=2048
    sr = 48000

    # 데이터 전처리 
    data_folder = 'midi_2_wav/samples'
    batch_size = 2
    dataset = SpectrogramDataset(data_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 모델 및 Trainer 생성
    model = Autoencoder()
    trainer = L.Trainer(max_epochs=100)

    # 학습 시작
    trainer.fit(model, dataloader)
 