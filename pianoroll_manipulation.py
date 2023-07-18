## midi -> pianoroll

from miditoolkit.midi import parser as mid_parser
from miditoolkit.pianoroll import parser as pr_parser
from miditoolkit.pianoroll import utils


# load midi file
path_midi = './generated_midi/practice.midi'
mido_obj = mid_parser.MidiFile(path_midi)
notes = mido_obj.instruments[0].notes

# convert notes to pianoroll
pianoroll = pr_parser.notes2pianoroll(
                    notes)

# pianoroll: tick x pitch
print("pianoroll shape :  ", pianoroll.shape)

# cropping pitch
pianoroll_small = pianoroll[:, 24:108]
print('before :           ', pianoroll_small.shape)

# padding pitch
pianoroll_re = utils.pitch_padding(pianoroll_small, pitch_range=(24, 108))
print('after :            ', pianoroll_re.shape)

# convert to chromagram
chormagram = utils.tochroma(pianoroll)
print("chromagram shape : ", chormagram.shape)

# resampling
old_resol = mido_obj.ticks_per_beat
new_resol = 24
factor_down = new_resol/ old_resol
print('resampling factor :', factor_down)

# downsample
pianoroll = pr_parser.notes2pianoroll(
                    notes, 
                    resample_factor=factor_down)
print('downsample :       ', pianoroll.shape)

# upsample
notes_re = pr_parser.pianoroll2notes(
                    pianoroll,
                    resample_factor=1/factor_down)
# check
for pair in zip(notes_re[:5],  notes[:5]):
    print('{} | {}'.format(pair[0], pair[1]))
