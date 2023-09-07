from miditoolkit.midi import parser as mid_parser
from miditoolkit.pianoroll import parser as pr_parser
from miditoolkit.pianoroll import utils
import numpy as np

# load midi file
path_midi = 'midi_2_wav/drum_data_practice/generated_midi/hh_16/hh_16_0.midi'
mido_obj = mid_parser.MidiFile(path_midi)
notes = mido_obj.instruments[0].notes

# resampling
old_resol = mido_obj.ticks_per_beat # 480
new_resol = 24
factor_down =  1 #new_resol/ old_resol
print('resampling factor:', factor_down)

# downsample
pianoroll = pr_parser.notes2pianoroll(
                    notes, 
                    resample_factor=factor_down)
print('pianoroll shape :', pianoroll.shape)

# breakpoint()
# upsample
notes_re = pr_parser.pianoroll2notes(
                    pianoroll,
                    resample_factor=1/factor_down)
# check
for pair in zip(notes_re[:5],  notes[:5]):
    print('{} | {}'.format(pair[0], pair[1]))

print("painoroll shape :  " ,pianoroll.size)

if isinstance(pianoroll, np.ndarray):
    print("The variable is a NumPy array.")
else:
    print("The variable is not a NumPy array.")