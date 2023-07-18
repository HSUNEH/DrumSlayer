import numpy as np

from miditoolkit.midi import parser as mid_parser  
from miditoolkit.midi import containers as ct

import os
######################
# generating 4tick (1beat)
##### parameter #####
midi_number = 10
note_number = 4
midi_name   = 'snare'
sample_rate = 48000
#####################
# velocity range: (0,127)
velocity_mu    = 100
velocity_sigma = 3
# pitch range : (0,127) 
pitch_mu    = 60
pitch_sigma = 3
#####################

for _ in range(midi_number):
    # create an empty file
    mido_obj = mid_parser.MidiFile()
    beat_resol = mido_obj.ticks_per_beat

    # create an  instrument
    track = ct.Instrument(program=0, is_drum=True, name='example track')
    mido_obj.instruments = [track]

    # 1beat = 1920 / 1tick = 480
    midi_numpy = np.zeros([2,sample_rate * 8]) # sample rate * sec

    for i in range(note_number):
        # create one note
        start = np.random.randint(0, 1919)
        end = start + 1

        velocity = int(np.random.normal(velocity_mu, velocity_sigma))
        pitch = int(np.random.normal(pitch_mu, pitch_sigma))
        
        note = ct.Note(start=start, end=end, pitch=pitch, velocity=velocity)

        mido_obj.instruments[0].notes.append(note)
                
        # midi to numpy
        # 1sec -> 48000 
        # 1 tick -> 480, 2sec -> 960000 sample 
        start_n = start*200
        midi_numpy[0,start_n] = (velocity+1)/128
        midi_numpy[1,start_n] = pitch

    # create markers
    marker_hi = ct.Marker(time=0, text='HI')
    mido_obj.markers.append(marker_hi)

    # write to file
    output_dir = f'midi_2_wav/drum_data_practice/generated_midi/{midi_name}'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{midi_name}_{_}.midi')
    mido_obj.dump(output_file)

    # write to midi file
    output_dir = f'midi_2_wav/drum_data_practice/generated_midi_numpy/{midi_name}'
    os.makedirs(output_dir, exist_ok=True)
    np.save(f'midi_2_wav/drum_data_practice/generated_midi_numpy/{midi_name}_{_}', midi_numpy)    



    # # reload for check
    # mido_obj_re = mid_parser.MidiFile(output_file)
    # for note in mido_obj_re.instruments[0].notes:
    #     print(note)


    # print('\nmarker:', mido_obj_re.markers)