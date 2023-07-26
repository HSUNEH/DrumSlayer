import os
import librosa
import numpy as np
from torch.utils.data import Dataset
from natsort import natsorted
from scipy import signal


class Loop(Dataset):
    def __init__(self, single_shot_dataset, midi_dataset, loop_seconds, data_length, reference_pitch=48):
        self.ssdataset = single_shot_dataset
        self.mididataset = midi_dataset
        self.loop_length = loop_seconds * self.ssdataset.sample_rate
        self.length = data_length
        self.reference_pitch = reference_pitch
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Choose a random dataset
        ssindex = np.random.choice(len(self.ssdataset))
        midiindex = index

        # Get singleshot & velocity & pitch
        ss = self.ssdataset[ssindex]
        velocity, pitch = self.mididataset[midiindex]
        onset = (velocity != 0).astype(int)
        pitch_shifted = (pitch - self.reference_pitch)*onset

        # Make a list consisted of velocity and pitch
        velocity_and_pitch = []
        for pitch in np.unique(pitch_shifted):
            mask = pitch_shifted == pitch
            velocity_masked = velocity * mask
            velocity_and_pitch.append([velocity_masked, pitch])

        # Make a loop
        loop = np.zeros((2, self.loop_length))
        for velocity, pitch in velocity_and_pitch:
            if pitch != 0:
                ss_shifted = librosa.effects.pitch_shift(ss, sr=self.ssdataset.sample_rate, n_steps=pitch)
            else:
                ss_shifted = ss
            loop += np.array([signal.convolve(velocity, ss_shifted[0])[:self.loop_length], 
                            signal.convolve(velocity, ss_shifted[1])[:self.loop_length]])
            
        return loop, velocity, pitch


class SingleShot(Dataset):
    def __init__(self, directory, sample_rate):
        self.sample_rate = sample_rate
        self.data = [os.path.join(directory, f) for f in natsorted(os.listdir(directory)) if f.endswith('.wav')]
        # data = directory내 wav파일 list
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio, _ = librosa.load(self.data[idx], sr=self.sample_rate, mono=False)
        if audio.ndim == 1:
            audio = np.stack((audio, audio))
        return audio
    

class MIDI(Dataset):
    def __init__(self, directory):
        self.data = [os.path.join(directory, f) for f in natsorted(os.listdir(directory)) if f.endswith('.npy')]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return np.load(self.data[idx])


if __name__ == '__main__':
    from tqdm import tqdm
    import soundfile as sf

    sample_rate = 48000
    loop_seconds = 2

    dir_ss_kick = './midi_2_wav/drum_data_practice/proprietary_dataset/kick'
    dir_ss_snare = './midi_2_wav/drum_data_practice/proprietary_dataset/snare'
    dir_ss_hhclosed = './midi_2_wav/drum_data_practice/proprietary_dataset/hhclosed'
    dir_ss_hhopen = './midi_2_wav/drum_data_practice/proprietary_dataset/hhopen'

    dir_midi_kick = './midi_2_wav/drum_data_practice/generated_midi_numpy/kick'
    dir_midi_snare = './midi_2_wav/drum_data_practice/generated_midi_numpy/snare'
    dir_midi_hhclosed = './midi_2_wav/drum_data_practice/generated_midi_numpy/hihat'

    ss_kick = SingleShot(dir_ss_kick, sample_rate)
    ss_snare = SingleShot(dir_ss_snare, sample_rate)
    ss_hhclosed = SingleShot(dir_ss_hhclosed, sample_rate)
    ss_hhopen = SingleShot(dir_ss_hhopen, sample_rate)

    midi_kick = MIDI(dir_midi_kick)
    midi_snare = MIDI(dir_midi_snare)
    midi_hhclosed = MIDI(dir_midi_hhclosed)

    loop_kick = Loop(ss_kick, midi_kick, loop_seconds, 100)
    loop_snare = Loop(ss_snare, midi_snare, loop_seconds, 100)
    loop_hhclosed = Loop(ss_hhclosed, midi_hhclosed, loop_seconds, 100)

    for idx in tqdm(range(10)):
        audio_loop_kick, _, _  = loop_kick[idx]
        audio_loop_snare, _, _  = loop_snare[idx]
        audio_loop_hhclosed, _, _  = loop_hhclosed[idx]
        audio_loop_drum = audio_loop_kick + audio_loop_snare + audio_loop_hhclosed
        audio_loop_drum = np.transpose(audio_loop_drum)
        output_dir = './midi_2_wav/samples/'
        os.makedirs(output_dir, exist_ok=True)
        sf.write( f'{output_dir}'+f'{idx}.wav', audio_loop_drum, sample_rate)