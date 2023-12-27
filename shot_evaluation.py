import librosa
import numpy as np
from pystoi import stoi
from pesq import pesq
from python_speech_features import mfcc
from scipy.spatial.distance import euclidean
from scipy.signal import stft

import argparse

#pip install librosa numpy pystoi pesq python_speech_features scipy

def calculate_mse(y1, y2):
    return np.mean((y1 - y2) ** 2)

def calculate_lsd(y1, y2, sr):
    f1, t1, Zxx1 = stft(y1, fs=sr)
    f2, t2, Zxx2 = stft(y2, fs=sr)
    return np.mean(np.sqrt(np.mean(np.square(np.abs(np.log10(np.abs(Zxx1) + 1e-10) - np.log10(np.abs(Zxx2) + 1e-10))), axis=0)))

def calculate_mcd(y1, y2, sr):
    mfcc1 = mfcc(y1, sr)
    mfcc2 = mfcc(y2, sr)
    return np.mean([euclidean(mfcc1[i], mfcc2[i]) for i in range(min(len(mfcc1), len(mfcc2)))])

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--oneshot_dir', type=str, default='./midi_2_wav/one_shots/', help='input data directory')
    # parser.add_argument('--output_dir', type=str, default='./generated_data/', help='output data directory')
    # args = parser.parse_args()


    # Load the audio files
    original_wav_dir = 'data_generate/midi_2_wav/one_shots/train/kick/01_DNB_Kicks.wav' #args.oneshot_dir
    reconstructed_wav_dir = 'data_generate/midi_2_wav/one_shots/train/kick/01_DNB_Kicks.wav' #args.output_dir

    y1, sr1 = librosa.load(original_wav_dir, sr=None)
    y_1, sr_1 = librosa.load(original_wav_dir, sr=16000)

    y2, sr2 = librosa.load(reconstructed_wav_dir, sr=None)
    y_2, sr_2 = librosa.load(reconstructed_wav_dir, sr=16000)


    print(sr1)
    print(sr2)
    print(sr_1)
    print(sr_2)

    # Ensure same sampling rate
    if sr1 != sr2:
        raise ValueError("Sampling rates are different.")

    # Pad the shorter file
    max_len = max(len(y1), len(y2))
    y1 = np.pad(y1, (0, max_len - len(y1)), 'constant')
    y2 = np.pad(y2, (0, max_len - len(y2)), 'constant')
    # Pad the shorter file
    max_len = max(len(y1), len(y2))
    y_1 = np.pad(y_1, (0, max_len - len(y_1)), 'constant')
    y_2 = np.pad(y_2, (0, max_len - len(y_2)), 'constant')

    # Calculate metrics
    mse = calculate_mse(y1, y2)
    # pesq_score = pesq(sr_1, y_1, y_2, 'wb')
    stoi_score = stoi(y1, y2, sr1, extended=False)
    lsd = calculate_lsd(y1, y2, sr1)
    mcd = calculate_mcd(y1, y2, sr1)

    print(f"MSE: {mse}")
    # print(f"PESQ: {pesq_score}")
    print(f"STOI: {stoi_score}")
    print(f"LSD: {lsd}")
    print(f"MCD: {mcd}")


