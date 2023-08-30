# TODO 1) audio -> Log mel spectrogram -> encoding -> decoding -> MIDI file
# Baseline 1) : Encoder Decoder (Onset + DrummerNet)
## Loss
# Onset : 
# DrummerNet : Mean Absolute Error between audio CQT

# TODO 2): MIDI file  <-> midi token 
# Baseline 2) : MT3 - T5 transformer 


# audiopreprocessing :  audio -> log mel
# model :               encoding & decoding


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import lightning as L
from audiopreprocessing import audio_to_log_mel_spectrogram
import librosa
import numpy as np
import os

import wandb

from torchmetrics.classification import MulticlassF1Score

# class T5(L.LightningModule):
#     def __init__(self):

#     def forward(self, x):

#     def training_step(self, batch, batch_idx):
    
#     def configure_optimizers(self):

class Autoencoder(L.LightningModule):
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
        self.last_conv = nn.Conv1d(100, 100, 3, padding = 1, stride=1)

        self.hidden_size = 3
        self.num_layers = 3 # 악기 개수
        self.channel_size = 100 # input / 16
        self.bi_lstm = nn.LSTM(self.channel_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)

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
            # print("d_conv : ", x.shape)
            xs.append(x)
        
        y = self.act(self.encode_conv(xs.pop()))
        
        ys = []
        ys.append(y)
        for module in self.u_convs:
            y = module(y)
            y = self.act(y)
            # print('before interpolate : ',y.shape)
            y = F.interpolate(y, scale_factor=2,
                    mode=int(y.dim() == 4) * 'bi' + 'linear', align_corners=False)
            x = xs.pop()

            y = torch.cat((y, x), dim=1)
            # print("u_conv : ", y.shape)
            ys.append(y)
        
        r = self.act(self.last_conv(y)) # (batch, 100, rate/16)
        # print("r : ", r.shape)
        r = r[...,:-80]
        r = r.transpose(1,2)
        # print("r (tr): ", r.shape)
        
        h0 = torch.zeros(self.num_layers*2, r.size(0), self.hidden_size).to(r.device) # Hidden state
        c0 = torch.zeros(self.num_layers*2, r.size(0), self.hidden_size).to(r.device) # Cell state

        after, (hn, cn) = self.bi_lstm(r, (h0,c0)) 

        # print("lstm : ", after.shape)
        
        represent = after.transpose(1,2)

        # print("lstm(tr) : ", represent.shape)

        result_np = torch.zeros([represent.size(0),6,96000], dtype=torch.float32) # TODO : batch, end값 삽입
        # for k in range(batch_size):
        #     for i in range(6):
        #         for j in range(96000//16):
        #             result_np[k][i][j*16] = represent[k][i][j]
        upsample_factor = 96000 // 6000  # 확장 비율 계산
        for i in range(upsample_factor):
            result_np[:, :, i::upsample_factor] = represent
        # print("result : " , result_np.shape)
        return result_np

    def training_step(self, batch, batch_idx):
        # import pdb ; pdb.set_trace()
        x, y_target = batch
        # batch -> torch.Size([batch_size, 96768]) torch.Size([batch_size, 6, 96000])
        y_target = y_target.float()
        y_pred = self(x).to('cuda') #torch.Size([batch_size, 6, 96000])
        criterion  = nn.MSELoss()
        loss = criterion(y_pred, y_target)  
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        # wandb.log({"train_loss": loss})  # Log loss to wandb
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_target = batch
        # batch -> torch.Size([batch_size, 96768]) torch.Size([batch_size, 6, 96000])
        y_target = y_target.float()
        y_pred = self(x).to('cuda') #torch.Size([batch_size, 6, 96000])
        
        criterion  = nn.MSELoss()
        loss = criterion(y_pred, y_target)  
        
        # metric = BinaryF1Score()
        # f1score = metric(y_pred, y_target)

        metrics = {"val_loss": loss}#, "val_f1score": f1score}
        self.log_dict(metrics , prog_bar=True, on_step=True, on_epoch=False)
        # wandb.log(metrics)  # Log loss to wandb
        return metrics

    # def test_step(self, batch, batch_idx):
    #     x, y_target = batch
    #     # batch -> torch.Size([batch_size, 96768]) torch.Size([batch_size, 6, 96000])
    #     y_target = y_target.float()
    #     y_pred = self(x).to('cuda') #torch.Size([batch_size, 6, 96000])
    
    #     criterion  = nn.MSELoss()
    #     loss = criterion(y_pred, y_target)  
        
    #     metric = BinaryF1Score()
    #     f1score = metric(y_pred, y_target)

    #     metrics = {"val_loss": loss, "val_f1score": f1score}
    #     self.log_dict(metrics , prog_bar=True, on_step=True, on_epoch=False)
        # wandb.log(metrics)  # Log loss to wandb
    #     return metrics
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

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
        if spectrogram.shape[0] % 1024 != 0:
            padding = spectrogram.shape[0] % 1024
            spectrogram = torch.cat([spectrogram, torch.zeros(padding)])
        # print("spectrogram : ",spectrogram.shape)


        kick_np = np.load(os.path.join(self.kick_nps_folder, self.kick_nps[idx]))
        kick_torch = torch.from_numpy(kick_np)
        hihat_np = np.load(os.path.join(self.hihat_nps_folder, self.hihat_nps[idx]))
        hihat_torch = torch.from_numpy(hihat_np)
        snare_np = np.load(os.path.join(self.snare_nps_folder, self.snare_nps[idx]))
        snare_torch = torch.from_numpy(snare_np)

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
        self.df = os.path.join(self.data_folder,'drum_data')
        self.dataset = SpectrogramDataset(self.df)
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers = 16)

  
    def val_dataloader(self):
        self.df = os.path.join(self.data_folder,'drum_data_val')
        self.dataset = SpectrogramDataset(self.df)
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers = 16)
  
    # def test_dataloader(self):
    #     self.df = os.path.join(self.data_folder,'drum_data_test')
    #     self.dataset = SpectrogramDataset(self.df)
    #     return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers = 16)
  
from lightning.pytorch.callbacks import ModelCheckpoint




if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project='Beat2Midi')
    top_models_callback = ModelCheckpoint(
    dirpath="model",
    verbose=True,
    every_n_train_steps=100,
    save_top_k=-1,
    save_last=True,
    filename="model-{epoch:02d}-{val_loss:.2f}",
    )

    # Parameters
    n_mels=128
    hop_length=512 
    n_fft=2048
    sr = 48000
    
    # 데이터 전처리 
    batch_size = 32
    dataloader = DrumTorchDataset('midi_2_wav', batch_size)
    # val_dataloader = DrumTorchDataset('midi_2_wav/drum_data_val', batch_size)

    # 모델 및 Trainer 생성
    model = Autoencoder().to('cuda')
    trainer = L.Trainer(accelerator="gpu", devices= 2, max_epochs=10, strategy="ddp", callbacks=[top_models_callback])

    # 학습 시작
    trainer.fit( model, datamodule = dataloader)#, val_dataloader)
