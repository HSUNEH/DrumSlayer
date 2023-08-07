# audio -> Log mel spectrogram

import librosa
import numpy as np

def audio_to_log_mel_spectrogram(audio_path, n_mels=128, hop_length=512, n_fft=2048, sr = 48000):
    # Load audio file
    y, sr = librosa.load(audio_path, sr = sr)
    # print(y)
    # print(sr)
    # Compute mel spectrogram
    # mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)

    # Convert to log scale (decibels)
    # log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return y #log_mel_spectrogram

if __name__ == "__main__":

    audio_path = "midi_2_wav/drum_data/samples/0.wav"  # Replace with the path to your audio file
    log_mel_spec = audio_to_log_mel_spectrogram(audio_path)

    print(log_mel_spec.shape)
#     # Display the log mel spectrogram
#     import librosa.display
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(log_mel_spec, x_axis='time', y_axis='mel', sr=sr, hop_length=hop_length)
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Log Mel Spectrogram')
#     plt.tight_layout()
#     plt.show()
