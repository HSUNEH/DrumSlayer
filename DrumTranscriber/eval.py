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

def ksh_generate(test_dataloader):
    for idx, batch in tqdm(enumerate(test_dataloader)):
        x, y = batch # x : torch.Size([1, 345, 2, 9]) , y : torch.Size([1, 9, 1063])
        # batch = x.cuda(7) , y.cuda(7)
        end = False
        while end == False: #end token이 나올 때까지 반복
            y_pred, end = model(batch) # torch.Size([1, 1062, 9]), False
            # for i in range(9):
            if y_pred[0,0,0] == 0 or y_pred[0,0,0] == 3:
                continue
            else:
                print('start token fucked')
                end = False
            for j in [1,2]:
                if y_pred[0,354*j,0] != 3:
                    print(f'{j}th sep token fucked')
                    end = False
            if end == False:
                print('regenerate')
            else:
                print('well generated')

        kick_tokens = y_pred[:,1+0:354,:]          #torch.Size([1, 353, 9])
        snare_tokens = y_pred[:,1+354:354*2,:]     #torch.Size([1, 353, 9])
        hihat_tokens = y_pred[:,1+354*2:354*3,:]   #torch.Size([1, 353, 9])
        

        
        seq_len = 345
        for i in range(9):
            # TODO: remove the max function later.
            kick_tokens[:, :seq_len, i] = kick_tokens[:, i:i+seq_len, i] - 4
            snare_tokens[:, :seq_len, i] = snare_tokens[:, i:i+seq_len, i] - 4
            hihat_tokens[:, :seq_len, i] = hihat_tokens[:, i:i+seq_len, i] - 4
            # audio_tokens[:, :seq_len, i] = torch.min(torch.ones(1).long().cuda()*1024,audio_tokens[:, i:i+seq_len, i])
        # kick_tokens[kick_tokens < 0] = 0
        # snare_tokens[snare_tokens < 0] = 0
        # hihat_tokens[hihat_tokens < 0] = 0
        kick_tokens = kick_tokens[:, :seq_len, :]   #torch.Size([1, 345, 9])
        snare_tokens = snare_tokens[:, :seq_len, :] #torch.Size([1, 345, 9])
        hihat_tokens = hihat_tokens[:, :seq_len, :] #torch.Size([1, 345, 9])

        # -2 값이 나오는 부분 전까지의 인덱스 찾기
        try:
            kick_end_index = torch.where(kick_tokens == -2)[1][0]
            kick_tensor = kick_tokens[:, :kick_end_index, :]
        except:
            kick_tensor = kick_tokens

        try:
            snare_end_index = torch.where(snare_tokens == -2)[1][0]
            snare_tensor = snare_tokens[:, :snare_end_index, :]
        except:
            snare_tensor = snare_tokens
        
        try:
            hihat_end_index = torch.where(hihat_tokens == -2)[1][0]
            hihat_tensor = hihat_tokens[:, :hihat_end_index, :]
        except:
            hihat_tensor = hihat_tokens

            
        # slicing하여 [1(batch_size), seq_len, 9] 형태로 만들기
        
        
        
        
        for i, codes in enumerate([kick_tensor, snare_tensor, hihat_tensor]):
            inst_codes = codes.permute(0,2,1) # torch.Size([1, 9, seq_len])
            latent = dac_model.quantizer.from_codes(inst_codes)[0]
            audio = dac_model.decode(latent)[0]
            audio = audio.detach().cpu().numpy().astype(np.float32)
            if i == 0:
                output_dir = result_dir + f'{idx}_kick.wav'
            elif i==1:
                output_dir = result_dir + f'{idx}_snare.wav'
            else:
                output_dir = result_dir + f'{idx}_hihat.wav'
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            scipy.io.wavfile.write(output_dir, 44100, audio.T)

        
        # kick_name = load_dac('/disk2/st_drums/generated_data/drum_data_test/kickShotList.txt', idx)
        # snare_name = load_dac('/disk2/st_drums/generated_data/drum_data_test/snareShotList.txt', idx)
        # hihat_name = load_dac('/disk2/st_drums/generated_data/drum_data_test/hihatShotList.txt', idx)
        
        destination_dir = './results/'
        
        # kick_file = '/disk2/st_drums/one_shots/test/kick/'+ kick_name
        # new_filename = f'{idx}_kick_t.wav'
        # shutil.copy(kick_file, os.path.join(destination_dir, new_filename))

        # snare_file = '/disk2/st_drums/one_shots/test/snare/'+ snare_name
        # new_filename = f'{idx}_snare_t.wav'
        # shutil.copy(snare_file, os.path.join(destination_dir, new_filename))
        
        # hihat_file = '/disk2/st_drums/one_shots/test/hhclosed/'+ hihat_name
        # new_filename = f'{idx}_hihat_t.wav'
        # shutil.copy(hihat_file, os.path.join(destination_dir, new_filename))
  

def inst_generate(test_dataloader, inst):

    for idx, batch in tqdm(enumerate(test_dataloader)):
        x, y, batch_length  = batch # x : torch.Size([1, 345, 2, 9]) , y : torch.Size([1, 9, seq_len])
        batch = x.cuda() , y.cuda(), batch_length
        end = False
        y_target = rearrange(y[:,:,:], 'b t d -> b d t')
        # y_pred, end, loss, padding_losses, padding_in_losses, audio_losses = model(batch) # torch.Size([1, seq_len, 9]), False
        y_pred, end = model(batch) # torch.Size([1, seq_len, 9]), False
        y_pred = y_pred[:,346:,:]
        # Audio part only -> inst_tokens
        try:
            end_token_idx = torch.where(y_pred == 0)[1][0]
            inst_tokens = y_pred[:,:end_token_idx,:]          #torch.Size([1, 353, 9])
        except:
            end_token_idx = y_pred.shape[1]
            inst_tokens = y_pred[:,:,:]

        inst_tokens = inst_tokens[:, :end_token_idx, :]   #torch.Size([1, 345, 9])
        inst_tokens = inst_tokens - 1



        # slicing하여 [1(batch_size), seq_len, 9] 형태로 만들기

        for _, codes in enumerate([inst_tokens]):
            print(codes.min(), codes.max())
            inst_codes = codes.permute(0,2,1) # torch.Size([1, 9, seq_len])
            latent = dac_model.quantizer.from_codes(inst_codes)[0]
            audio = dac_model.decode(latent)['audio']
            audio = audio.detach().cpu().numpy().astype(np.float32)

            output_dir = result_dir + f'{inst}/{idx}_{inst}.wav'

            os.makedirs(os.path.dirname(output_dir), exist_ok=True) #(1, 18432)
            scipy.io.wavfile.write(output_dir, 44100, audio.T)


        inst_name = load_dac(f'/data5/drumslayer/drum_data_{data_type}/{inst}ShotList.txt', idx)
        
        destination_dir = f'./results/{inst}/'
        inst_file = f'/data5/drumslayer/{data_type}/kick/'+ inst_name
        new_filename = f'{idx}_{inst}_t.wav'
        shutil.copy(inst_file, os.path.join(destination_dir, new_filename))

        # breakpoint()


# def main(ckpt_dir, data_dir, result_dir,audio_encoding_type, args):
#     config = EncoderDecoderConfig(audio_rep = audio_encoding_type, args = args)
#     module = EncoderDecoderModule.load_from_checkpoint(ckpt_dir, config=config)
#     trainer = pl.Trainer(accelerator="gpu", devices=NUM_DEVICES)

#     model = module.encoder_decoder
    
#     test_dataset = DrumSlayerDataset(data_dir, "test", audio_encoding_type, args)
#     test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
#     trainer.predict(model, dataloaders=test_dataloader)


    

# def evaluate(model, test_dataset, result_dir, args):
    
    
        
#     with torch.no_grad():
#         for i, batch in enumerate(test_dataloader):
#             x, y = batch

#             x = x.cuda()
#             # y_hat = model(x, strategy="greedy")
#             y_hat = model(x, strategy="top-p", sample_arg=0.5)

#             y = y.detach().numpy()
#             midi_gt = convert_to_midi(y)
#             midi_gt.write(f"results/{i:03d}_gt.mid")
#             plot_midi(midi_gt, f"results/{i:03d}_gt.png")

#             y_hat = y_hat.cpu().detach().numpy()
#             midi_pred = convert_to_midi(y_hat)
#             midi_pred.write(f"results/{i:03d}_pred.mid")
#             plot_midi(midi_pred, f"results/{i:03d}_pred.png")

#             audio = af.read(data_dir + f"drum_data_test/dafx_loops/{i}.wav")[0]
#             af.write(f"results/{i:03d}.wav", audio, 44100)
#             import pdb; pdb.set_trace()
        
# def convert_to_midi(tokens):
#     tokens = tokens[0,1:-1]
#     num_notes = tokens.shape[0] // 3
#     drum_midi = pretty_midi.PrettyMIDI()
#     drum_inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
#     type_to_pitch = [36, 38, 42]
#     for i in range(num_notes):
#         onset = tokens[i*3] - 3
#         vel = tokens[i*3+1] - 3 - 1000
#         type = tokens[i*3+2] - 3 - 1000 - 128
#         if type not in [0, 1, 2]:
#             break
#         drum_inst.notes.append(pretty_midi.Note(
#             velocity=vel,
#             pitch=type_to_pitch[type],
#             start=onset*0.005,
#             end=(onset+1)*0.005
#         ))
#     drum_midi.instruments.append(drum_inst)
#     return drum_midi

# def plot_midi(midi, fname):
#     drum_roll = draw_drum_roll(midi.instruments[0].notes)
#     plt.figure(figsize=(12, 4))
#     plt.imshow(drum_roll, aspect='auto', interpolation='nearest')
#     plt.savefig(fname)




if __name__ == "__main__":
    
    data_dir = '/disk2/st_drums/generated_data/'
    result_dir = './results/'
    audio_encoding_type = 'codes'
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_type', type=str, default='kick', help='ksh, kshm, kick, snare, hihat')
    parser.add_argument('--wandb', type=bool, default='False', help='True, False')
    parser.add_argument('--layer_cut', type=int, default='4', help='enc(or dec)_num_layers // layer_cut')
    parser.add_argument('--dim_cut', type=int, default='4', help='enc(or dec)_num_heads, _d_model // dim_cut')
    parser.add_argument('--batch_size', type=int, default='1', help='batch size')
    args = parser.parse_args()
    
    if args.train_type == 'ksh':
        ckpt_dir = '/workspace/DrumTranscriber/ckpts/2024-03-12-05-STDT-ksh-2_2_4/epoch=0-train_total_loss=0.14.ckpt'
    elif args.train_type == 'kick':
        # ckpt_dir = '/workspace/DrumTranscriber/ckpts/2024-03-07-09-STDT-kick-2_2_4/epoch=0-train_total_loss=0.14.ckpt'
        ckpt_dir = '/workspace/DrumTranscriber/ckpts/03-26-10-31-STDT-kick-1_1_3/train_total_loss=0.13-valid_total_loss=0.05.ckpt'
        ckpt_dir = '/home/kyungsukim/git/DrumSlayer/DrumTranscriber/ckpt/03-27-00-24-STDT-kick-4_4_3/last.ckpt.ckpt'
    
    dac_model = DAC()
    dac_model = load_model(dac.__model_version__)
    dac_model.eval()
    dac_model.cuda()
    
    # config = EncoderDecoderConfig(audio_rep = audio_encoding_type, args = args)
    # model = EncoderDecoderModule(config)
    config = InstDecoderConfig(audio_rep = audio_encoding_type, args = args)
    model = InstDecoderModule(config)
    
    ckpt = torch.load(ckpt_dir, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.cuda()
    model.eval()
    # device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    data_type = "debug"
    test_dataset = DrumSlayerDataset(data_dir, data_type, audio_encoding_type, args)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0) # x: audio dac, y : target token 

    if args.train_type == 'ksh':
        ksh_generate(test_dataloader)
    elif args.train_type == 'kick' or args.train_type == 'snare' or args.train_type == 'hihat':
        inst_generate(test_dataloader, args.train_type)
    
