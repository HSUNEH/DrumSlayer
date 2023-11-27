import numpy as np
from DAFX import DrumChains

# mono, sample rate
mono = False
sample_rate = 44100

# define chain
drumchains = DrumChains(mono, sample_rate)

# prepare kick, hihat, snare loops. should be numpy array!
kick_loop = np.random.rand(2, 44100*10)
hihat_loop = np.random.rand(2, 44100*10)
snare_loop = np.random.rand(2, 44100*10)

# make DAFxed drum mix
drum_mix_mastered = drumchains.apply(kick_loop, snare_loop, hihat_loop)
