import numpy as np
from DAFX import DrumChains, MasteringChains, InstChains
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

class OtherLoop(Dataset):
    def __init__(self, loop_2sec_dataset, loop_4sec_dataset, loop_seconds,  midi_num): #60
        self.loop2s_dataset = loop_2sec_dataset
        self.loop4s_dataset = loop_4sec_dataset
        self.loop_seconds = loop_seconds
        self.loop_length = loop_seconds * self.loop2s_dataset.sample_rate
        self.midi_num = midi_num
    def __len__(self):
        return len(self.midi_num)

    def __getitem__(self,idx):
        loop_4s_num = int(self.loop_seconds // 4)  # 4sec 개수
        loop_left = int(self.loop_seconds % 4)          
        loop_2s_num = int(loop_left // 2)               # 2sec 개수        
        loop_left = int(loop_left % 2)                  # 1sec 유무 (0 or 1)

        loop = np.array([[],[]]) #stereo
        
        # stack 4sec loops
        for n in range(loop_4s_num):
            # Choose 4s dataset randomly
            loop4s_index = np.random.choice(len(self.loop4s_dataset))
            loop4s, loop_name = self.loop4s_dataset[loop4s_index]
            # pitch shift randomly
            loop4s = librosa.effects.pitch_shift(loop4s, sr = self.loop4s_dataset.sample_rate, n_steps=np.random.randint(-12, 12))
            # padding or slicing
            if loop4s.shape[1] < (self.loop4s_dataset.sample_rate * 4):
                loop4s = np.pad(loop4s, ((0,0),(0,self.loop4s_dataset.sample_rate * 4 - loop4s.shape[1])), mode='constant')
            else:
                loop4s = loop4s[:, :self.loop4s_dataset.sample_rate * 4]
            loop = np.concatenate((loop, loop4s), axis=1)
            
        # stack 2sec loops  
        for n in range(loop_2s_num):
            # Choose 2s dataset randomly
            loop2s_index = np.random.choice(len(self.loop2s_dataset))
            loop2s, loop_name = self.loop2s_dataset[loop2s_index]
            # pitch shift randomly
            loop2s = librosa.effects.pitch_shift(loop2s, sr = self.loop2s_dataset.sample_rate, n_steps=np.random.randint(-12, 12))
            # padding or slicing
            if loop2s.shape[1] < (self.loop2s_dataset.sample_rate * 2):
                loop2s = np.pad(loop2s, ((0,0),(0,self.loop2s_dataset.sample_rate * 2 - loop2s.shape[1])), mode='constant')
            else:
                loop2s = loop2s[:, :self.loop2s_dataset.sample_rate * 2]
            loop = np.concatenate((loop, loop2s), axis=1)

        # stack 1sec loops (randomly slice from 2s, 4s dataset)
        if loop_left == 1:
            # Randomly choose 2s or 4s dataset
            loop2s_or_4s = np.random.choice([2, 4])
            if loop2s_or_4s == 2:
                # Choose 2s dataset randomly
                loop2s_index = np.random.choice(len(self.loop2s_dataset))
                loop2s, loop_name = self.loop2s_dataset[loop2s_index]
                loop1s = loop2s[:, :self.loop2s_dataset.sample_rate]
            else:
                # Choose 4s dataset randomly
                loop4s_index = np.random.choice(len(self.loop4s_dataset))
                loop4s, loop_name = self.loop4s_dataset[loop4s_index]
                loop1s = loop4s[:, :self.loop2s_dataset.sample_rate]
            # pitch shift randomly
            loop1s = librosa.effects.pitch_shift(loop1s, sr = self.loop4s_dataset.sample_rate, n_steps=np.random.randint(-12, 12))
            loop = np.concatenate((loop, loop1s), axis=1)

        return loop
    
class OtherSound(Dataset):
    def __init__(self, directory, sample_rate):
        self.sample_rate = sample_rate
        self.data = [os.path.join(directory, f) for f in natsorted(os.listdir(directory)) if f.endswith('.wav')]
        # self.data = directory내 wav파일 list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio, _ = librosa.load(self.data[idx], sr=self.sample_rate, mono=False)
        if audio.ndim == 1:
            audio = np.stack((audio, audio)) # mono to stereo
        return audio, self.data[idx]

def generate_drum_fx(args):
    data_type = args.data_type    
    if data_type == 'all':
        drum_fx_all(args)
    else:
        drum_fx_one(args)
    return None

def generate_drum_other_fx(kick,snare,hihat, piano, guitar, bass, args):
    # mono, sample rate
    mono = args.mono
    sample_rate = args.sample_rate
    output_dir = args.output_dir
    
    # define chain
    drumchains = DrumChains(mono, sample_rate)

    masteringchains = MasteringChains(mono, sample_rate)
    # make DAFXed drum mix
    kick_modified, snare_modified, hihat_modified = drumchains.apply(kick, snare, hihat)
    drum_mix_mastered = masteringchains.apply(kick_modified, snare_modified, hihat_modified)

    return drum_mix_mastered

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
    