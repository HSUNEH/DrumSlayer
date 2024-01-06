# TODO: Mixed Audio -> DAC -> transformer -> DAC (kick, snare, hihat) ->  wav output
# import pytorch lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


import argparse

import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# For reproducibility.
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


class Transformer(nn.Module):

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def encode(self, x):
        out = self.encoder(x)
        return out


    def decode(self, z, c):
        out = self.decode(z, c)
        return out


    def forward(self, x, z):
        c = self.encode(x)
        y = self.decode(z, c)
        return y



if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='all', help='all, train, valid, test')
    args = parser.parse_args()
