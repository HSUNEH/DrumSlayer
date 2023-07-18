import numpy as np

from miditoolkit.midi import parser as mid_parser  
from miditoolkit.midi import containers as ct

# load midi file
path_midi = './generated_midi/hihat_9.midi'
midi_obj = mid_parser.MidiFile(path_midi)
print(midi_obj)
# ticks per beats
print(' > ticks per beat:', midi_obj.ticks_per_beat)

# signatures
print('\n -- time signatures -- ')
print(' > amount:', len(midi_obj.time_signature_changes))
print(midi_obj.time_signature_changes[0])

print('\n -- key signatures --')
print(' > amount:', len(midi_obj.key_signature_changes))

# tempo changes
print('\n -- tempo changes --')
print(' > amount:', len(midi_obj.tempo_changes))
print(midi_obj.tempo_changes[0])

# markers
# print('\n -- markers --')
# print(' > amount:', len(midi_obj.markers))
# print(midi_obj.markers[0])

# instruments
# print('\n -- instruments --')
# print(' > number of tracks:', len(midi_obj.instruments))
# print(' > number of notes:', len(midi_obj.instruments[0].notes))

for note in midi_obj.instruments[0].notes:
    print(note)

# convert to seconds
note = midi_obj.instruments[0].notes[7]
mapping = midi_obj.get_tick_to_time_mapping()
tick = note.start
sec = mapping[tick]
print('{} tick at {} seconds'.format(tick, sec))