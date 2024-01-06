import os
import librosa
import numpy as np
from torch.utils.data import Dataset
from natsort import natsorted
from scipy import signal
import torch
import sys
from scipy.io.wavfile import write

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from DAFXChain.drum_fx import generate_drum_other_fx

class OtherLoop(Dataset):
    def __init__(self, loop_2sec_dataset, loop_4sec_dataset, loop_seconds): #60
        self.loop2s_dataset = loop_2sec_dataset
        self.loop4s_dataset = loop_4sec_dataset
        self.loop_seconds = loop_seconds
        self.loop_length = loop_seconds * self.loop2s_dataset.sample_rate
    def __len__(self):
        return 1

    def __getitem__(self,inst):
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
            if inst == 'bass':
                loop4s = librosa.effects.pitch_shift(loop4s, sr = self.loop4s_dataset.sample_rate, n_steps=np.random.randint(-6, 2))
            else:
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
            if inst == 'bass':
                loop2s = librosa.effects.pitch_shift(loop2s, sr = self.loop2s_dataset.sample_rate, n_steps=np.random.randint(-6, 2))
            else:
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
            if inst == 'bass':
                loop1s = librosa.effects.pitch_shift(loop1s, sr = self.loop4s_dataset.sample_rate, n_steps=np.random.randint(-6, 2))
            else:
                loop1s = librosa.effects.pitch_shift(loop1s, sr = self.loop4s_dataset.sample_rate, n_steps=np.random.randint(-12, 12))
            loop = np.concatenate((loop, loop1s), axis=1)

        return loop

class VocalLoop(Dataset):
    def __init__(self, loop_vocal_dataset, loop_seconds): #60
        self.loop_vocal_dataset = loop_vocal_dataset
        self.loop_seconds = loop_seconds
        self.loop_length = loop_seconds * self.loop_vocal_dataset.sample_rate
    def __len__(self):
        return 1

    def __getitem__(self,inst):

        loop = np.array([[],[]]) #stereo
        
        # 1. choose sample ramdomly
        loop_index = np.random.choice(len(self.loop_vocal_dataset))
        vocal_loop, loop_name = self.loop_vocal_dataset[loop_index]
        
        # 2. sample length
        if vocal_loop.shape[1] > self.loop_length:
            #randomly slice vocal_loop
            start = np.random.randint(0, vocal_loop.shape[1] - self.loop_length)
            loop = vocal_loop[:, start:start+self.loop_length]
        else:
            # loop_length가 채워질 때 까지 vocal_loop을 반복해서 붙임
            loop = np.tile(vocal_loop, (1, self.loop_length // vocal_loop.shape[1] + 1))
            loop = loop[:, :self.loop_length]

        # 3. pitch shift randomly
        loop = librosa.effects.pitch_shift(loop, sr = self.loop_vocal_dataset.sample_rate, n_steps=np.random.randint(-12,12))


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

class Loop(Dataset):
    def __init__(self, single_shot_dataset, midi_dataset, loop_seconds, output_dir, data_type, inst, reference_pitch=48, render_type='slice'): #60
        self.ssdataset = single_shot_dataset
        self.mididataset = midi_dataset
        self.loop_length = loop_seconds * self.ssdataset.sample_rate
        self.reference_pitch = reference_pitch
        self.output_dir = output_dir
        self.data_type = data_type
        self.inst = inst
        self.render_type = render_type
    def __len__(self):
        return len(self.mididataset)

    def __getitem__(self, index):
        # Choose a random dataset
        ssindex = np.random.choice(len(self.ssdataset))
        midiindex = index

        # Get singleshot & velocity & pitch
        ss, ss_name = self.ssdataset[ssindex]
        ss_name = os.path.basename(ss_name)
        # singleshot list 저장
        ss_file = self.output_dir + f'drum_data_{self.data_type}/{self.inst}ShotList.txt'
        if midiindex == 0:
            np.savetxt(ss_file, np.reshape(ss_name, (1,)), fmt='%s')
        else:
            with open(ss_file, 'a') as f:
                np.savetxt(f, np.reshape(ss_name, (1,)), fmt='%s')


        velocity, pitch = self.mididataset[midiindex] # MIDI dataset returns velocity and pitch, but we don't use pitch for drum loop dataset.
        loop = np.zeros((2, self.loop_length))

        if self.render_type == 'convolution':
            loop = np.array([signal.convolve(velocity, ss[0])[:self.loop_length], 
                             signal.convolve(velocity, ss[1])[:self.loop_length]])

        elif  self.render_type == 'slice':
            onset_idx_list = velocity.nonzero()[0].tolist()
            for i, onset_idx in enumerate(onset_idx_list):
                # Calculate the interval between current and next onset. 
                if i< len(onset_idx_list) -1: 
                    interval = onset_idx_list[i+1] - onset_idx
                else:
                    interval = self.loop_length - onset_idx

                # Place the singleshot at the onset position.
                if interval < ss.shape[1]:
                    loop[:, onset_idx:onset_idx+interval] +=  velocity[onset_idx] * ss[:, :interval]
                else:
                    loop[:, onset_idx:onset_idx+ss.shape[1]] += velocity[onset_idx] * ss

        return loop, velocity, pitch


class SingleShot(Dataset):
    def __init__(self, directory, sample_rate):
        self.sample_rate = sample_rate
        self.data = [os.path.join(directory, f) for f in natsorted(os.listdir(directory)) if f.endswith('.wav')]
        # self.data = directory내 wav파일 list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio, _ = librosa.load(self.data[idx], sr=self.sample_rate, mono=False)
        if audio.ndim == 1:
            audio = np.stack((audio, audio)) # make mono to stereo
        return audio, self.data[idx]


class MIDI(Dataset):
    def __init__(self, directory, sample_rate, loop_seconds):
        self.data = [os.path.join(directory, f) for f in natsorted(os.listdir(directory)) if f.endswith('.npy')]
        self.sr = sample_rate
        self.loop_seconds = loop_seconds
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # breakpoint()
        midi_960n = np.load(self.data[idx]) # (2, 960*loop_seconds)
        midi_sr_n = np.zeros([midi_960n.shape[0], self.sr * self.loop_seconds]) # (2, sample_rate * loop_second)
        
        for i in range(midi_960n.shape[0]):
            for j in range(midi_960n.shape[1]):
                midi_sr_n[i, int(round(j*self.sr/960))] = midi_960n[i, j]
        
        return midi_sr_n


def generate_midi_2_wav(args):
    data_type = args.data_type
    if args.other_sounds:
        midi_2_wav_other_all(args)
    else:    
        if data_type == 'all':
            midi_2_wav_all(args)
        else:
            midi_2_wav_one(args)
    return None

def midi_2_wav_other_all(args):
    from tqdm import tqdm
    import soundfile as sf
    import torch
    all = ['train', 'valid', 'test']
    midi_number = args.midi_number
    loop_seconds = args.loop_seconds
    midi_numbers = [int(midi_number*0.9), int(midi_number*0.05), int(midi_number*0.05)]
    for data_type in all:
        
        sample_rate = args.sample_rate
        loop_seconds = args.loop_seconds
        oneshot_dir = args.oneshot_dir
        output_dir = args.output_dir
        render_type = args.render_type

        #single shot dir
        dir_ss_kick =       oneshot_dir + f'{data_type}/kick'
        dir_ss_snare =      oneshot_dir + f'{data_type}/snare'
        dir_ss_hhclosed =   oneshot_dir + f'{data_type}/hhclosed'
        # dir_ss_hhopen = './midi_2_wav/drum_data_practice/proprietary_dataset/hhopen'

        #midi numpy dir
        dir_midi_kick =     output_dir + f'drum_data_{data_type}/generated_midi_numpy/kick_midi_numpy'
        dir_midi_snare =    output_dir + f'drum_data_{data_type}/generated_midi_numpy/snare_midi_numpy'
        dir_midi_hhclosed = output_dir + f'drum_data_{data_type}/generated_midi_numpy/hihat_midi_numpy'

        ss_kick = SingleShot(dir_ss_kick, sample_rate)
        ss_snare = SingleShot(dir_ss_snare, sample_rate)
        ss_hhclosed = SingleShot(dir_ss_hhclosed, sample_rate)
        # ss_hhopen = SingleShot(dir_ss_hhopen, sample_rate)

        midi_kick = MIDI(dir_midi_kick, sample_rate, loop_seconds)
        midi_snare = MIDI(dir_midi_snare,sample_rate, loop_seconds)
        midi_hhclosed = MIDI(dir_midi_hhclosed,sample_rate, loop_seconds) 

        loop_kick = Loop(ss_kick, midi_kick, loop_seconds, output_dir, data_type, 'kick')
        loop_snare = Loop(ss_snare, midi_snare, loop_seconds, output_dir, data_type, 'snare')
        loop_hhclosed = Loop(ss_hhclosed, midi_hhclosed, loop_seconds, output_dir, data_type, 'hihat')

        loop_piano_2sec = OtherSound(oneshot_dir + f'piano/2sec' , sample_rate)
        loop_piano_4sec = OtherSound(oneshot_dir + f'piano/4sec' , sample_rate)
        loop_guitar_2sec = OtherSound(oneshot_dir + f'guitar/2sec' , sample_rate)
        loop_guitar_4sec = OtherSound(oneshot_dir + f'guitar/4sec' , sample_rate)
        loop_bass_2sec = OtherSound(oneshot_dir + f'bass/2sec' , sample_rate)
        loop_bass_4sec = OtherSound(oneshot_dir + f'bass/4sec' , sample_rate)
        
        sample_vocal = OtherSound(oneshot_dir + f'vocals' , sample_rate)
        # TODO : vocal cut by loop_seconds

        loop_piano = OtherLoop(loop_piano_2sec,loop_piano_4sec,loop_seconds)
        loop_guitar = OtherLoop(loop_guitar_2sec,loop_guitar_4sec,loop_seconds )
        loop_bass = OtherLoop(loop_bass_2sec,loop_bass_4sec,loop_seconds)

        loop_vocal = VocalLoop(sample_vocal, loop_seconds)

        # Bring the each loop separately
        for idx in tqdm(range(len(loop_kick)), desc=f'midi2wav {data_type} data'): 
            audio_loop_kick, _, _  = loop_kick[idx]
            audio_loop_snare, _, _  = loop_snare[idx]
            audio_loop_hhclosed, _, _  = loop_hhclosed[idx]

            audio_loop_piano = loop_piano['piano']
            audio_loop_guitar = loop_guitar['guitar']
            audio_loop_bass = loop_bass['bass']
            audio_loop_vocal = loop_vocal['vocal']

            # loop + DAFX => mixed_loop
            mixed_loop = generate_drum_other_fx(audio_loop_kick, audio_loop_snare, audio_loop_hhclosed, audio_loop_piano, audio_loop_guitar, audio_loop_bass, audio_loop_vocal,args)
            
            # numpy to wav write
            mixed_loop_dir = output_dir + f'drum_data_{data_type}/mixed_loops/'
            os.makedirs(mixed_loop_dir, exist_ok=True)
            write(mixed_loop_dir + f'{idx}.wav', sample_rate, mixed_loop.T)

    return None    

def midi_2_wav_all(args):
    from tqdm import tqdm
    import soundfile as sf
    import torch
    all = ['train', 'valid', 'test']
    for data_type in all:
        
        sample_rate = args.sample_rate
        loop_seconds = args.loop_seconds
        oneshot_dir = args.oneshot_dir
        output_dir = args.output_dir
        render_type = args.render_type

        #single shot dir
        dir_ss_kick =       oneshot_dir + f'{data_type}/kick'
        dir_ss_snare =      oneshot_dir + f'{data_type}/snare'
        dir_ss_hhclosed =   oneshot_dir + f'{data_type}/hhclosed'
        # dir_ss_hhopen = './midi_2_wav/drum_data_practice/proprietary_dataset/hhopen'

        #midi numpy dir
        dir_midi_kick =     output_dir + f'drum_data_{data_type}/generated_midi_numpy/kick_midi_numpy'
        dir_midi_snare =    output_dir + f'drum_data_{data_type}/generated_midi_numpy/snare_midi_numpy'
        dir_midi_hhclosed = output_dir + f'drum_data_{data_type}/generated_midi_numpy/hihat_midi_numpy'

        ss_kick = SingleShot(dir_ss_kick, sample_rate)
        ss_snare = SingleShot(dir_ss_snare, sample_rate)
        ss_hhclosed = SingleShot(dir_ss_hhclosed, sample_rate)
        # ss_hhopen = SingleShot(dir_ss_hhopen, sample_rate)

        midi_kick = MIDI(dir_midi_kick, sample_rate, loop_seconds)
        midi_snare = MIDI(dir_midi_snare,sample_rate, loop_seconds)
        midi_hhclosed = MIDI(dir_midi_hhclosed,sample_rate, loop_seconds) 

        loop_kick = Loop(ss_kick, midi_kick, loop_seconds, output_dir, data_type, 'kick', render_type=render_type)
        loop_snare = Loop(ss_snare, midi_snare, loop_seconds, output_dir, data_type, 'snare', render_type=render_type)
        loop_hhclosed = Loop(ss_hhclosed, midi_hhclosed, loop_seconds, output_dir, data_type, 'hihat', render_type=render_type)

        # Bring the each loop separately
        for idx in tqdm(range(len(loop_kick)), desc=f'midi2wav {data_type} data'): 
            audio_loop_kick, _, _  = loop_kick[idx]
            audio_loop_snare, _, _  = loop_snare[idx]
            audio_loop_hhclosed, _, _  = loop_hhclosed[idx]
            
            audio_loop_kick = np.transpose(audio_loop_kick)
            audio_loop_snare = np.transpose(audio_loop_snare)
            audio_loop_hhclosed = np.transpose(audio_loop_hhclosed)
            
            # save generated_loops
            kick_dir = output_dir + f'drum_data_{data_type}/generated_loops/kick/'
            os.makedirs(kick_dir, exist_ok=True)
            snare_dir = output_dir + f'drum_data_{data_type}/generated_loops/snare/'
            os.makedirs(snare_dir, exist_ok=True)
            hhclosed_dir = output_dir + f'drum_data_{data_type}/generated_loops/hhclosed/'
            os.makedirs(hhclosed_dir, exist_ok=True)

            sf.write( f'{kick_dir}'+f'{idx}.wav', audio_loop_kick, sample_rate)
            sf.write( f'{snare_dir}'+f'{idx}.wav', audio_loop_snare, sample_rate)
            sf.write( f'{hhclosed_dir}'+f'{idx}.wav', audio_loop_hhclosed, sample_rate)

    return None

# TODO: Need to update filenames, arguments to function call, etc.
def midi_2_wav_one(args):
    from tqdm import tqdm
    import soundfile as sf
    import torch
    data_type = args.data_type
    sample_rate = args.sample_rate
    loop_seconds = args.loop_seconds
    oneshot_dir = args.oneshot_dir
    output_dir = args.output_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #single shot dir
    dir_ss_kick =       oneshot_dir + f'{data_type}/kick'
    dir_ss_snare =      oneshot_dir + f'{data_type}/snare'
    dir_ss_hhclosed =   oneshot_dir + f'{data_type}/hhclosed'
    # dir_ss_hhopen = './midi_2_wav/drum_data_practice/proprietary_dataset/hhopen'

    #midi numpy dir
    dir_midi_kick =     output_dir + f'drum_data_{data_type}/generated_midi_numpy/kick_midi_numpy'
    dir_midi_snare =    output_dir + f'drum_data_{data_type}/generated_midi_numpy/snare_midi_numpy'
    dir_midi_hhclosed = output_dir + f'drum_data_{data_type}/generated_midi_numpy/hhclosed_midi_numpy'

    ss_kick = SingleShot(dir_ss_kick, sample_rate)
    ss_snare = SingleShot(dir_ss_snare, sample_rate)
    ss_hhclosed = SingleShot(dir_ss_hhclosed, sample_rate)
    # ss_hhopen = SingleShot(dir_ss_hhopen, sample_rate)

    midi_kick = MIDI(dir_midi_kick, sample_rate, loop_seconds)
    midi_snare = MIDI(dir_midi_snare,sample_rate, loop_seconds)
    midi_hhclosed = MIDI(dir_midi_hhclosed,sample_rate, loop_seconds) 

    loop_kick = Loop(ss_kick, midi_kick, loop_seconds)
    loop_snare = Loop(ss_snare, midi_snare, loop_seconds)
    loop_hhclosed = Loop(ss_hhclosed, midi_hhclosed, loop_seconds)

    # Bring the each loop separately
    for idx in tqdm(range(len(loop_kick)), desc=f'midi2wav {data_type} data'): 
        audio_loop_kick, _, _  = loop_kick[idx]
        audio_loop_snare, _, _  = loop_snare[idx]
        audio_loop_hhclosed, _, _  = loop_hhclosed[idx]
        
        audio_loop_kick = np.transpose(audio_loop_kick)
        audio_loop_snare = np.transpose(audio_loop_snare)
        audio_loop_hhclosed = np.transpose(audio_loop_hhclosed)
        
        kick_dir = f'./generated_data/drum_data_{data_type}/generated_loops/kick/'
        os.makedirs(kick_dir, exist_ok=True)
        snare_dir = f'./generated_data/drum_data_{data_type}/generated_loops/snare/'
        os.makedirs(snare_dir, exist_ok=True)
        hhclosed_dir = f'./generated_data/drum_data_{data_type}/generated_loops/hhclosed/'
        os.makedirs(hhclosed_dir, exist_ok=True)

        sf.write( f'{kick_dir}'+f'{idx}.wav', audio_loop_kick, sample_rate)
        sf.write( f'{snare_dir}'+f'{idx}.wav', audio_loop_snare, sample_rate)
        sf.write( f'{hhclosed_dir}'+f'{idx}.wav', audio_loop_hhclosed, sample_rate)
        
    return None

    # # Bring the whole loop at once
    # for idx in tqdm(range(len(loop_kick))): 
    #     audio_loop_kick, _, _  = loop_kick[idx]
    #     audio_loop_snare, _, _  = loop_snare[idx]
    #     audio_loop_hhclosed, _, _  = loop_hhclosed[idx]
    #     audio_loop_drum = audio_loop_kick + audio_loop_snare + audio_loop_hhclosed
    #     audio_loop_drum = np.transpose(audio_loop_drum)
    #     output_dir = f'./generated_data/drum_data_{data_type}/generated_loops/'
    #     os.makedirs(output_dir, exist_ok=True)
    #     sf.write( f'{output_dir}'+f'{idx}.wav', audio_loop_drum, sample_rate)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='all', help='all, train, valid, test')
    parser.add_argument('--other_sounds', type=bool, default=True, help='other sounds')
    parser.add_argument('--sample_rate', type=int, default=44100, help='sample_rate')
    parser.add_argument('--loop_seconds', type=int, default=5, help='loop_seconds')
    parser.add_argument('--oneshot_dir', type=str, default='./one_shots/', help='input data directory')
    parser.add_argument('--output_dir', type=str, default='../generated_data/', help='output data directory')
    parser.add_argument('--midi_number', type=int, default=20, help='midi number')
    parser.add_argument('--mono', type=bool, default=False, help='mono or stereo')
    args = parser.parse_args()
    generate_midi_2_wav(args)