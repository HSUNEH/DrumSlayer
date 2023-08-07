import numpy as np

from miditoolkit.midi import parser as mid_parser  
from miditoolkit.midi import containers as ct
from tqdm import tqdm
import os
######################

##### parameter #####
midi_number = 1000        # midi 개수
note_number = 2         # onset 개수
midi_name   = 'snare_16r'

bpm = 120
sec_per_tick = 60 / bpm
tick_number = 4         # 4tick = 1beat
print("Midi sec : ", sec_per_tick * tick_number )
sample_rate = 48000         
#####################
# mu, sigma -> musig = True
# musig = True
# velocity range: (0,127)
velocity_mu    = 100
velocity_sigma = 3
# pitch range : (0,127) 
pitch_mu    = 60
pitch_sigma = 2

# random -> musig = False
musig = False
vel_range = [95,105] #[64,128]
pitch_range = [48,72]
#####################
grid_music = False # start on ( beat (1920) / note number )
grid_music_musig = True
grid_music_sigma = 10

grid_16 = True
#####################


for _ in tqdm(range(midi_number)):
    # create an empty file
    mido_obj = mid_parser.MidiFile()
    beat_resol = mido_obj.ticks_per_beat

    # create an  instrument
    track = ct.Instrument(program=0, is_drum=True, name = midi_name)
    mido_obj.instruments = [track]

    # 1beat = 1920 / 1tick = 480
    midi_numpy = np.zeros([2,sample_rate * 8]) # sample rate * sec

    for i in range(note_number):
        # create one note
        if grid_music == True:     # 1beat = 1920 / 1tick = 480 (1beat = 4tick)
            if grid_music_musig == True:
                if i == 0 :
                    start = abs(int(tick_number * 480 / note_number) * i + int(np.random.normal(0, grid_music_sigma)))
                else:
                    start = int(tick_number * 480 / note_number) * i + int(np.random.normal(0, grid_music_sigma))
                end = start + 1
            else:
                start = int(tick_number * 480 / note_number) * i 
                end = start + 1
        elif grid_16 == True:
            start = np.random.randint(0, (tick_number * 16))
            start *= 30
            if start == 480*tick_number:
                start -= 1
            end = start + 1
        else: 
            start = np.random.randint(0, (tick_number * 480) -1)
            end = start + 1

        if musig == True:
            velocity = int(np.random.normal(velocity_mu, velocity_sigma))
            pitch = int(np.random.normal(pitch_mu, pitch_sigma))
        else:
            velocity = np.random.randint(vel_range[0],vel_range[1])
            pitch = np.random.randint(pitch_range[0],pitch_range[1])

        note = ct.Note(start=start, end=end, pitch=pitch, velocity=velocity)

        mido_obj.instruments[0].notes.append(note)

        start_n = int(start*(sec_per_tick * tick_number * sample_rate /(tick_number*480)))
        midi_numpy[0,start_n] = (velocity+1)/128
        midi_numpy[1,start_n] = pitch

    # create markers
    marker_hi = ct.Marker(time=0, text='HI')
    mido_obj.markers.append(marker_hi)

    # write to file
    output_dir = f'midi_2_wav/drum_data/generated_midi/{midi_name}'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{midi_name}_{_}.midi')
    mido_obj.dump(output_file)

    # write to midi file
    output_dir = f'midi_2_wav/drum_data/generated_midi_numpy/{midi_name}'
    os.makedirs(output_dir, exist_ok=True)
    np.save(output_dir+f'/{midi_name}_{_}', midi_numpy)    



    # # reload for check
    # mido_obj_re = mid_parser.MidiFile(output_file)
    # for note in mido_obj_re.instruments[0].notes:
    #     print(note)


    # print('\nmarker:', mido_obj_re.markers)
print(f"generated '{midi_number}' number of '{midi_name}' succeed")