import numpy as np
from DAFX import DrumChains, AllMasteringChains, MasteringChains, InstChains
import argparse
from audiotools import AudioSignal
from scipy.io.wavfile import write
import shutil
import os
from tqdm import tqdm
import librosa
import numpy as np
from torch.utils.data import Dataset
from natsort import natsorted
from scipy import signal


def generate_drum_fx(args):
    data_type = args.data_type    
    if data_type == 'all':
        drum_fx_all(args)
    else:
        drum_fx_one(args)
    return None

def generate_drum_other_fx(kick, snare, hihat, piano, guitar, bass, args):
    # mono, sample rate
    mono = args.mono
    sample_rate = args.sample_rate
    output_dir = args.output_dir
    
    # define chain
    drumchains = DrumChains(mono, sample_rate)
    instchains = InstChains(mono, sample_rate)
    masteringchains = AllMasteringChains(mono, sample_rate)

    # make DAFXed drum mix
    kick_modified, snare_modified, hihat_modified = drumchains.apply(kick, snare, hihat)
    piano_modified, guitar_modified, bass_modified = instchains.apply(piano, guitar, bass)

    mix_mastered_loop = masteringchains.apply(kick_modified, snare_modified, hihat_modified, piano_modified, guitar_modified, bass_modified)

    return mix_mastered_loop

def drum_fx_all(args):
    # mono, sample rate
    for num, data_type in enumerate(['train', 'valid', 'test']):
        midi_number = args.midi_number
        midi_numbers = [int(midi_number*0.9), int(midi_number*0.05), int(midi_number*0.05)]
        mono = args.mono
        sample_rate = args.sample_rate
        output_dir = args.output_dir
        # define chain
        drumchains = DrumChains(mono, sample_rate)
        masteringchains = MasteringChains(mono, sample_rate)


        for i in tqdm(range(midi_numbers[num]), desc=f'DAFX {data_type} data'):
            # prepare kick, hihat, snare loops. should be numpy array!
            kick = AudioSignal(output_dir + f'drum_data_{data_type}/generated_loops/kick/{i}.wav').numpy().squeeze()
            snare = AudioSignal(output_dir + f'drum_data_{data_type}/generated_loops/snare/{i}.wav').numpy().squeeze()
            hihat = AudioSignal(output_dir + f'drum_data_{data_type}/generated_loops/hhclosed/{i}.wav').numpy().squeeze() #(2, 220500)

            # make DAFXed drum mix
            kick_modified, snare_modified, hihat_modified = drumchains.apply(kick, snare, hihat)

            # mastering
            drum_mix_mastered = masteringchains.apply(kick_modified, snare_modified, hihat_modified)

            # numpy to wav write
            dafx_loop_dir = output_dir + f'drum_data_{data_type}/dafx_loops/'
            os.makedirs(dafx_loop_dir, exist_ok=True)
            write(dafx_loop_dir + f'{i}.wav', sample_rate, drum_mix_mastered.T)



    return None

def drum_fx_one(args):
    # mono, sample rate
    mono = args.mono
    sample_rate = args.sample_rate
    output_dir = args.output_dir
    # define chain
    drumchains = DrumChains(mono, sample_rate)
    masteringchains = MasteringChains(mono, sample_rate)

    for i in tqdm(range(args.midi_number), desc=f'DAFX {args.data_type} data'):
        # prepare kick, hihat, snare loops. should be numpy array!
        kick = AudioSignal(output_dir + f'drum_data_{args.data_type}/generated_loops/kick/{i}.wav').numpy().squeeze()
        snare = AudioSignal(output_dir + f'drum_data_{args.data_type}/generated_loops/snare/{i}.wav').numpy().squeeze()
        hihat = AudioSignal(output_dir + f'drum_data_{args.data_type}/generated_loops/hhclosed/{i}.wav').numpy().squeeze()

        # make DAFxed drum mix
        kick_modified, snare_modified, hihat_modified = drumchains.apply(kick, snare, hihat)
        drum_mix_mastered = masteringchains.apply(kick_modified, snare_modified, hihat_modified)
        # numpy to wav write
        dafx_loop_dir = output_dir + f'drum_data_{args.data_type}/dafx_loops/'
        os.makedirs(dafx_loop_dir, exist_ok=True)
        write(dafx_loop_dir + f'{i}.wav', sample_rate, drum_mix_mastered.T)

    return None




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_rate', type=int, default=44100, help='sample_rate')
    parser.add_argument('--mono', type=bool, default=False, help='mono or stereo')
    parser.add_argument('--midi_number', type=int, default=10, help='midi number')
    parser.add_argument('--data_type', type=str, default='all', help='train, val, test')
    parser.add_argument('--oneshot_dir', type=str, default='/Users/hwang/DrumSlayer/data_generate/midi_2_wav/one_shots/', help='input data directory')
    parser.add_argument('--output_dir', type=str, default='/Users/hwang/DrumSlayer/data_generate/generated_data/', help='output data directory')
    parser.add_argument('--loop_seconds', type=int, default=5, help='loop_seconds')
    args = parser.parse_args()
    drum_fx_all(args)
    