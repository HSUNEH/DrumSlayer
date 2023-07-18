import numpy as np

from miditoolkit.midi import parser as mid_parser  
from miditoolkit.midi import containers as ct

# load midi file
path_midi = 'testcases/2.midi'
mido_obj = mid_parser.MidiFile(path_midi)
beat_resol = mido_obj.ticks_per_beat

# define interval: from 2nd to 8th bar
st = beat_resol * 4 * 2
ed = beat_resol * 4 * 8

# export
mido_obj.dump('seg.midi', segment=(st, ed))