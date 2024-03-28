from encoder_decoder_inst_c import EncoderDecoderModule, EncoderDecoderConfig    
from inst_decoder import InstDecoderModule, InstDecoderConfig
from dataset_c import DrumSlayerDataset
from torch.utils.data import DataLoader
import wandb
import lightning.pytorch as pl
import os
import audiofile as af
import pretty_midi
import matplotlib.pyplot as plt
from investigate import draw_drum_roll
import argparse
import torch
from tqdm import tqdm
import dac
from dac.utils import load_model
from dac.model import DAC
import scipy.io.wavfile
from einops import rearrange
import numpy as np
import os
import shutil
from dac.utils import load_model
from dac.model import DAC


def load_dac(file_path, idx):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        oneshot_name = lines[idx].strip()  # Get the content at index idx and remove leading/trailing whitespaces
    return oneshot_name


def inst_generate(test_dataloader, inst):

    for idx, batch in tqdm(enumerate(test_dataloader)):
        x, y, batch_length  = batch # x : torch.Size([1, 345, 2, 9]) , y : torch.Size([1, 9, seq_len])
        if gpu:
            batch = x.cuda() , y.cuda(), batch_length.cuda()
        end = False
        y_target = rearrange(y[:,:,:], 'b t d -> b d t')
        # y_pred, end, loss, padding_losses, padding_in_losses, audio_losses = model(batch) # torch.Size([1, seq_len, 9]), False
        y_pred, end = model(batch) # torch.Size([1, seq_len, 9]), False
        if dac_only:
            if not torch.all(y_pred[:,353,:] == 0):
                print('Start Token FUCKED')
            if not torch.all(y_pred[:,-1,:] == 0):
                print('End Token FUCKED')

            inst_tokens = y_pred[:,354:-1,:]
            # Audio part only -> inst_tokens
        else: 
            if not torch.all(y_pred[:,-1,:] == 0):
                print('End Token FUCKED')
            inst_tokens = y_pred[:,1:-1,:]

        
        inst_tokens = inst_tokens - 1

        dac_len = inst_tokens.shape[1]-8

        # interleaving pattern 풀기

        for i in range(1,9): # dac : (9, (max)431) / codes : (s, d)
            inst_tokens[:,:dac_len,i] = inst_tokens[:,i:i+dac_len,i] 

            
        inst_tokens = inst_tokens[:,:-8,:]

        for _, codes in enumerate([inst_tokens]):
            inst_codes = codes.permute(0,2,1) # torch.Size([1, 9, seq_len])
            latent = dac_model.quantizer.from_codes(inst_codes)[0]
            audio = dac_model.decode(latent)[0]
            audio = audio.detach().cpu().numpy().astype(np.float32)

            output_dir = result_dir + f'{inst}/{idx}_{inst}.wav'

            os.makedirs(os.path.dirname(output_dir), exist_ok=True) #(1, 18432)
            scipy.io.wavfile.write(output_dir, 44100, audio.T)


        inst_name = load_dac(f'/disk2/st_drums/generated_data/drum_data_{data_type}/{inst}ShotList.txt', idx)
        
        destination_dir = f'/disk2/st_drums/results/{inst}/'
        inst_file = f'/disk2/st_drums/one_shots/{data_type}/kick/'+ inst_name
        new_filename = f'{idx}_{inst}_t.wav'
        shutil.copy(inst_file, os.path.join(destination_dir, new_filename))

        breakpoint()



if __name__ == "__main__":
    
    data_dir = '/disk2/st_drums/generated_data/'
    result_dir = '/disk2/st_drums/results/'
    audio_encoding_type = 'codes'
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_type', type=str, default='kick', help='ksh, kshm, kick, snare, hihat')
    parser.add_argument('--wandb', type=bool, default='False', help='True, False')
    parser.add_argument('--layer_cut', type=int, default='1', help='enc(or dec)_num_layers // layer_cut')
    parser.add_argument('--dim_cut', type=int, default='1', help='enc(or dec)_num_heads, _d_model // dim_cut')
    parser.add_argument('--batch_size', type=int, default='1', help='batch size')
    args = parser.parse_args()
    ######### MAIN #############
    
    
    
    dac_only = True
    gpu = False
    ckpt_dir = '/workspace/ckpts/03-28-08-12-STDT-kick-1_1_16/train_audio_loss=0.19-valid_audio_loss=2.34-step=4239.ckpt'
    #export CUDA_VISIBLE_DEVICES=2
    
    
    
    ############################
    dac_model_path = dac.utils.download(model_type="44khz")
    dac_model = dac.DAC.load(dac_model_path) 
    dac_model.eval()
    if gpu :
        dac_model.cuda()
    

    if dac_only:
        config = InstDecoderConfig(audio_rep = audio_encoding_type, args = args)
        model = InstDecoderModule(config)
    else:
        config = EncoderDecoderConfig(audio_rep = audio_encoding_type, args = args)
        model = EncoderDecoderModule(config)

    ckpt = torch.load(ckpt_dir, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    if gpu :
        model.cuda()
    model.eval()
    # device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    data_type = "debug"
    test_dataset = DrumSlayerDataset(data_dir, data_type, audio_encoding_type, args)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0) # x: audio dac, y : target token 

    
    inst_generate(test_dataloader, args.train_type)
    
    