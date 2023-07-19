# TODO : audio -> Log mel spectrogram -> encoding -> decoding -> MIDI file
# Baseline: Onset and frames 

# audiopreprocessing :  audio -> log mel
# model :               encoding & decoding


# TODO : pytorch lightning

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import lightning as L
from audiopreprocessing import audio_to_log_mel_spectrogram

class Autoencoder(L.LightningModule):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
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
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # Output should be between 0 and 1 (image pixel values)
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
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

audio_path = "path/to/your/audio_file.wav"  # Replace with the path to your audio file
log_mel_spec = audio_to_log_mel_spectrogram(audio_path)


# Parameters
n_mels=128
hop_length=512 
n_fft=2048
sr = 48000

# if __name__ == "__main__":
#     # 데이터 전처리
#     transform = transforms.Compose([transforms.ToTensor()])
#     mnist_train = MNIST('data', train=True, download=True, transform=transform)
#     mnist_train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

#     # 모델 및 Trainer 생성
#     model = Autoencoder()
#     trainer = L.Trainer(max_epochs=10, gpus=1)

#     # 학습 시작
#     trainer.fit(model, mnist_train_loader)
 