import numpy as np
from DAFX import DrumChains


# mono, sample rate
mono = False
sample_rate = 48000

# define chain
drumchains = DrumChains(mono, sample_rate)

# # prepare kick, hihat, snare loops. should be numpy array!
# kick_loop = np.random.rand(2, 44100*10)
# hihat_loop = np.random.rand(2, 44100*10)
# snare_loop = np.random.rand(2, 44100*10)

from audiotools import AudioSignal
kick_loop = AudioSignal("80kick.wav").numpy().squeeze()
snare_loop = AudioSignal("80snare.wav").numpy().squeeze()
hihat_loop = AudioSignal("80hat.wav").numpy().squeeze()


# make DAFxed drum mix
drum_mix_mastered = drumchains.apply(kick_loop, snare_loop, hihat_loop)

# numpy to wav write
from scipy.io.wavfile import write
write("drum_mix_mastered.wav", sample_rate, drum_mix_mastered.T)
