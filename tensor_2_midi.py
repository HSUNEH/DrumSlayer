# For test
from miditoolkit.midi import parser as mid_parser
from miditoolkit.midi import containers as ct
from miditoolkit.pianoroll import parser as pr_parser
from miditoolkit.pianoroll import utils
import numpy as np

######
path_np = "midi_2_wav/drum_data_test/generated_midi_numpy/hihat_16/hihat_16_0.npy" 
hihat_np = np.load(path_np) # TODO : (2,1920) np -> (1920,128) pianoroll
hihat_onset = hihat_np.nonzero()
hihat_pr = np.zeros((1920,128))
hihat_onset_num = len(hihat_onset[0])//2

# create an empty file
mido_obj = mid_parser.MidiFile()

# create an  instrument
track = ct.Instrument(program=0, is_drum=True, name='hihat')
mido_obj.instruments = [track]

# create notes
for i in range(hihat_onset_num):
    start = hihat_onset[1][i]
    end = start + 1
    pitch = int(hihat_np[1][hihat_onset[1][i]])
    velocity = int((hihat_np[0][hihat_onset[1][i]])*128)-1
    note = ct.Note(start=start, end=end, pitch=pitch, velocity=velocity)
    mido_obj.instruments[0].notes.append(note)

# create markers
marker_hi = ct.Marker(time=0, text='HI')
mido_obj.markers.append(marker_hi)

# write to file
mido_obj.dump('result.midi')

# load midi file
path_midi = 'result.midi'
mido_obj = mid_parser.MidiFile(path_midi)
notes = mido_obj.instruments[0].notes
print(notes)
print(hihat_onset)