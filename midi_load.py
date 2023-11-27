import numpy as np

from miditoolkit.midi import parser as mid_parser  
from miditoolkit.midi import containers as ct

# load midi file
path_midi = './00_--PJHxphWEs.mid'
mido_obj = mid_parser.MidiFile(path_midi)

# ticks per beats
print(' > ticks per beat:', mido_obj.ticks_per_beat)

# signatures
print('\n -- time signatures -- ')
print(' > amount:', len(mido_obj.time_signature_changes))
print(mido_obj.time_signature_changes[0])

print('\n -- key signatures --')
print(' > amount:', len(mido_obj.key_signature_changes))

# tempo changes
print('\n -- tempo changes --')
print(' > amount:', len(mido_obj.tempo_changes))
print(mido_obj.tempo_changes[0])

# markers
print('\n -- markers --')
print(' > amount:', len(mido_obj.markers))
print(mido_obj.markers[0])

# instruments
print('\n -- instruments --')
print(' > number of tracks:', len(mido_obj.instruments))
print(' > number of notes:', len(mido_obj.instruments[0].notes))

# convert to seconds
note = mido_obj.instruments[0].notes[20]
mapping = mido_obj.get_tick_to_time_mapping()
tick = note.start
sec = mapping[tick]
print('{} tick at {} seconds'.format(tick, sec))
