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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import lightning as L
from audiopreprocessing import audio_to_log_mel_spectrogram
import librosa
import numpy
import os


class Autoencoder():
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.compress_ratio = 2 ** 10 

        self.maxpool = nn.MaxPool1d(2)
        self.act = nn.ReLU()    

        self.d_conv0 = nn.Conv1d(1, 50, 3, padding = 1, stride = 1,bias=False)

        self.d_convs = nn.ModuleList(
            [nn.Conv1d(50, 50, 3, padding = 1,stride=1, bias = False) for _ in range(10)]
        )

        self.encode_conv = nn.Conv1d(50,50,3,padding = 1, stride= 1,bias=False)

        self.u_convs = nn.ModuleList([
            nn.Conv1d(50, 50, 3, padding = 1, stride=1, bias = False),
        ]+
        [nn.Conv1d(100, 50, 3, padding = 1, stride=1, bias = False) for _ in range(5)]
        )
        self.last_conv = nn.Conv1d(100, 100, 3, padding = 1, stride=1)
        
    def forward(self, x):
        div = self.compress_ratio
        nsp_src = x.shape[1]
        nsp_pad = (div - (nsp_src % div)) % div
        if nsp_pad != 0:
            x = F.pad(x, (0, nsp_pad))

        x = torch.unsqueeze(x,1)
        x= self.act(self.d_conv0(x)) # ( batch , n_ch , N) 
        
        
        xs = []
        for module in self.d_convs:
            x = module(x)
            x = self.act(x)
            x = self.maxpool(x)
            print(x.shape)
            xs.append(x)
        
        y = self.act(self.encode_conv(xs.pop()))
        
        ys = []
        ys.append(y)
        for module in self.u_convs:
            y = module(y)
            y = self.act(y)
            y = F.interpolate(y, scale_factor=2,
                    mode=int(y.dim() == 4) * 'bi' + 'linear', align_corners=False)
            x = xs.pop()

            y = torch.cat((y, x), dim=1)
            print(y.shape)
            ys.append(y)

        r = self.last_conv(y)
        print(r.shape)
        return self.act(r), xs, ys

    def training_step(self, batch, batch_idx):
        # batch -> torch.Size([batch_size, 128, 188])
        # TODO : batch에 target값도 같이 포함시켜 받기
        
        x = batch
        
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
        return torch.tensor(spectrogram, dtype=torch.float32)



if __name__ == "__main__":
    # Parameters
    n_mels=128
    hop_length=512 
    n_fft=2048
    sr = 48000

    # 데이터 전처리 
    data_folder = 'midi_2_wav/drum_data/samples'
    batch_size = 32
    dataset = SpectrogramDataset(data_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # torch([batch, 128, 188])

    # 모델 및 Trainer 생성
    model = Autoencoder()
    trainer = L.Trainer(max_epochs=10)

    # 학습 시작
    trainer.fit(model, dataloader)
    