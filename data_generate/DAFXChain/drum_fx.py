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
    def __init__(self, loop_2sec_dataset, loop_4sec_dataset, loop_seconds, output_dir, data_type, inst): #60
        self.loop2s_dataset = loop_2sec_dataset
        self.loop4s_dataset = loop_4sec_dataset
        self.loop_length = loop_seconds * self.loop2s_dataset.sample_rate
        self.output_dir = output_dir
        self.data_type = data_type
        self.inst = inst
    def __len__(self):
        return len(self.mididataset)

    def __getitem__(self, index):

        # Choose a random dataset
        ssindex = np.random.choice(len(self.loop2s_dataset))
        midiindex = index

        # Get singleshot & velocity & pitch
        ss, ss_name = self.loop2s_dataset[ssindex]
        ss_name = os.path.basename(ss_name)
        # TODO : singleshot 저장
        ss_file = self.output_dir + f'drum_data_{self.data_type}/{self.inst}ShotList.txt'
        if midiindex == 0:
            np.savetxt(ss_file, np.reshape(ss_name, (1,)), fmt='%s')
        else:
            with open(ss_file, 'a') as f:
                np.savetxt(f, np.reshape(ss_name, (1,)), fmt='%s')

        velocity, pitch = self.mididataset[midiindex]
        onset = (velocity != 0).astype(int)
        # pitch_shifted = (pitch - self.reference_pitch)*onset

        # Make a list consisted of velocity and pitch
        velocity_and_pitch = []
        # for pitch in np.unique(pitch_shifted):
        #     mask = pitch_shifted == pitch
        #     velocity_masked = velocity * mask
        #     velocity_and_pitch.append([velocity_masked, pitch])

        # Make a loop
        loop = np.zeros((2, self.loop_length))
        for velocity, pitch in velocity_and_pitch:
            if pitch != 0:
                ss_shifted = librosa.effects.pitch_shift(ss, sr=self.loop2s_dataset.sample_rate, n_steps=pitch)
            else:
                ss_shifted = ss
            loop += np.array([signal.convolve(velocity, ss_shifted[0])[:self.loop_length], 
                            signal.convolve(velocity, ss_shifted[1])[:self.loop_length]])

        return loop, velocity, pitch
    
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
            audio = np.stack((audio, audio))
        return audio, self.data[idx]

def generate_drum_fx(args):
    data_type = args.data_type
    if args.other_sounds:
        generate_drum_other_fx(args)
    else:    
        if data_type == 'all':
            drum_fx_all(args)
        else:
            drum_fx_one(args)
    return None

def generate_drum_other_fx(args):
    # mono, sample rate
    for num, data_type in enumerate(['train', 'valid', 'test']):
        midi_number = args.midi_number
        midi_number = [int(midi_number*0.9), int(midi_number*0.05), int(midi_number*0.05)]
        mono = args.mono
        sample_rate = args.sample_rate
        output_dir = args.output_dir
        oneshot_dir = args.oneshot_dir
        loop_seconds = args.loop_seconds
        # define chain
        drumchains = DrumChains(mono, sample_rate)
        masteringchains = MasteringChains(mono, sample_rate)

        loop_piano_2sec = OtherSound(oneshot_dir + f'piano/2sec' , sample_rate)
        loop_piano_4sec = OtherSound(oneshot_dir + f'piano/2sec' , sample_rate)
        loop_guitar_2sec = OtherSound(oneshot_dir + f'guitar/2sec' , sample_rate)
        loop_guitar_4sec = OtherSound(oneshot_dir + f'guitar/2sec' , sample_rate)
        loop_bass_2sec = OtherSound(oneshot_dir + f'bass/2sec' , sample_rate)
        loop_bass_4sec = OtherSound(oneshot_dir + f'bass/2sec' , sample_rate)

        loop_piano = OtherLoop(loop_piano_2sec,loop_piano_4sec,loop_seconds, output_dir, data_type, 'piano')

        for i in tqdm(range(midi_number[num]), desc=f'DAFX {data_type} data'):
            # prepare kick, hihat, snare loops. should be numpy array!
            kick = AudioSignal(output_dir + f'drum_data_{data_type}/generated_loops/kick/{i}.wav').numpy().squeeze()
            snare = AudioSignal(output_dir + f'drum_data_{data_type}/generated_loops/snare/{i}.wav').numpy().squeeze()
            hihat = AudioSignal(output_dir + f'drum_data_{data_type}/generated_loops/hhclosed/{i}.wav').numpy().squeeze()

            # make DAFXed drum mix
            kick_modified, snare_modified, hihat_modified = drumchains.apply(kick, snare, hihat)

            # TODO 

            drum_mix_mastered = masteringchains.apply(kick_modified, snare_modified, hihat_modified)

            # numpy to wav write
            dafx_loop_dir = output_dir + f'drum_data_{data_type}/dafx_loops/'
            os.makedirs(dafx_loop_dir, exist_ok=True)
            write(dafx_loop_dir + f'{i}.wav', sample_rate, drum_mix_mastered.T)


    return None

def drum_fx_all(args):
    # mono, sample rate
    for num, data_type in enumerate(['train', 'valid', 'test']):
        midi_number = args.midi_number
        midi_number = [int(midi_number*0.9), int(midi_number*0.05), int(midi_number*0.05)]
        mono = args.mono
        sample_rate = args.sample_rate
        output_dir = args.output_dir
        # define chain
        drumchains = DrumChains(mono, sample_rate)
        masteringchains = MasteringChains(mono, sample_rate)


        for i in tqdm(range(midi_number[num]), desc=f'DAFX {data_type} data'):
            # prepare kick, hihat, snare loops. should be numpy array!
            kick = AudioSignal(output_dir + f'drum_data_{data_type}/generated_loops/kick/{i}.wav').numpy().squeeze()
            snare = AudioSignal(output_dir + f'drum_data_{data_type}/generated_loops/snare/{i}.wav').numpy().squeeze()
            hihat = AudioSignal(output_dir + f'drum_data_{data_type}/generated_loops/hhclosed/{i}.wav').numpy().squeeze()

            # make DAFXed drum mix
            kick_modified, snare_modified, hihat_modified = drumchains.apply(kick, snare, hihat)

            # TODO 

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
    args = parser.parse_args()
    generate_drum_other_fx(args)
    