import numpy as np

from miditoolkit.midi import parser as mid_parser  
from miditoolkit.midi import containers as ct
from tqdm import tqdm
import os
import argparse



def generate_midi(args):
    data_type = args.data_type
    if data_type == 'all':
        generate_midi_all(args)
    else:
        generate_midi_one(args)
    return None

def generate_midi_all(args):
    # TODO : Bpm change
    # TODO : midi note 수가 Grid 보다 많을 때 raise error
    # TODO : 5 seconds 단위로 생성

    ##### parameter #####
    midi_number = args.midi_number    # midi 개수
    midi_n_list = [int(midi_number*0.9), int(midi_number*0.05), int(midi_number*0.05)]
    bpm = 120 # Sec 단위 생성이기 때문에 0.5 beat = 2 tick = 1 sec
    output_dir = args.output_dir

    sec_per_tick = 60 / bpm
    tick_number = 4*args.beat         # 4tick = 1beat
    print("Midi sec : ", sec_per_tick * tick_number )
    sample_rate = args.sample_rate         

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
    musig = True
    vel_range = [95,105] #[64,128]
    pitch_range = [48,72]

    #####################
    grid_music_sigma = 10
    #####################
    all = ['train', 'valid', 'test']
    for n,data_type in enumerate(all):
        midi_number = midi_n_list[n]

        for b in range(3):
            if b == 0:
                note_number = 8 *args.beat        # onset 개수
                midi_name = 'hihat_midi'          # 폴더,파일 이름
            elif b == 1:
                note_number = 2 *args.beat
                midi_name = 'kick_midi'
            elif b == 2 :
                note_number = 2 *args.beat
                midi_name = 'snare_midi'


            for _ in tqdm(range(midi_number)):
                # create an empty file
                mido_obj = mid_parser.MidiFile()
                beat_resol = mido_obj.ticks_per_beat

                # create an  instrument
                track = ct.Instrument(program=0, is_drum=True, name = midi_name)
                mido_obj.instruments = [track]

                # 1beat = 1920 / 1tick = 480
                midi_numpy = np.zeros([2,int(480 * tick_number)]) # midi_1920
                
                onsets = []
                for i in range(note_number):
                    # create one note

                    # onset(start)

                    if args.grid_random == 'R':
                        start = np.random.randint(0, (tick_number * 480) -1) # whole random
                        if start not in onsets:
                            onsets.append(start)
                        else:
                            while start in onsets:
                                start = np.random.randint(0, (tick_number * 480) -1)
                            onsets.append(start)
                    elif args.grid_random == 'G':
                        if b == 2: 
                            start = int(tick_number * 480 / note_number) * i + int(tick_number * 480 / note_number / 2)
                        else:
                            start = int(tick_number * 480 / note_number) * i 
                    elif args.grid_random == 'RG':
                        start = np.random.randint(0, (tick_number * 4)) # random in grid
                        start *= 120
                        if start == 480*tick_number:
                            start -= 1
                        if start not in onsets:
                            onsets.append(start)
                        else:
                            while start in onsets:
                                start = np.random.randint(0, (tick_number * 16))
                                start *= 30
                                
                            onsets.append(start)
                    elif args.grid_random == 'GG':
                        if i == 0 :
                            start = abs(int(tick_number * 480 / note_number) * i + int(np.random.normal(0, grid_music_sigma)))
                        else:
                            start = int(tick_number * 480 / note_number) * i + int(np.random.normal(0, grid_music_sigma)) 

                    if start == 480*tick_number:
                        start -= 1
                    end = start + 1

                    #velocity & pitch
                    if musig == True:
                        velocity = int(np.random.normal(velocity_mu, velocity_sigma))
                        pitch = 60
                        # pitch = int(np.random.normal(pitch_mu, pitch_sigma))
                    else:
                        velocity = np.random.randint(vel_range[0],vel_range[1])
                        pitch = 60
                        # pitch = np.random.randint(pitch_range[0],pitch_range[1])

                    note = ct.Note(start=start, end=end, pitch=pitch, velocity=velocity)

                    mido_obj.instruments[0].notes.append(note)

                    # start_n = int(start*(sec_per_tick * tick_number * sample_rate /(tick_number*480))) # start * 50 (1920 * 50  = 960000)
                    midi_numpy[0,start] = (velocity+1)/128
                    midi_numpy[1,start] = pitch

                # create markers
                marker_hi = ct.Marker(time=0, text='TT')
                mido_obj.markers.append(marker_hi)

                # write to file
                output_dir1 = output_dir + f'drum_data_{data_type}/generated_midi/{midi_name}'
                os.makedirs(output_dir1, exist_ok=True)
                output_file = os.path.join(output_dir1, f'{midi_name}_{_}.midi')
                mido_obj.dump(output_file)

                # write to midi file
                output_dir2 = output_dir + f'drum_data_{data_type}/generated_midi_numpy/{midi_name}_numpy'
                os.makedirs(output_dir2, exist_ok=True)
                np.save(output_dir2+f'/{midi_name}_{_}', midi_numpy)    


            print(f"generated '{midi_number}' number of '{midi_name}' succeed")
    return None  

def generate_midi_one(args):
    # TODO : Bpm change
    # TODO : midi note 수가 Grid 보다 많을 때 raise error
    # TODO : 5 seconds 단위로 생성

    ##### parameter #####
    midi_number = args.midi_number    # midi 개수
    data_type = args.data_type   # train, val, test, all
    bpm = 120 # Sec 단위 생성이기 때문에 0.5 beat = 2 tick = 1 sec
    output_dir = args.output_dir

    sec_per_tick = 60 / bpm
    tick_number = 4*args.beat         # 4tick = 1beat
    print("Midi sec : ", sec_per_tick * tick_number )
    sample_rate = args.sample_rate         

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
    musig = True
    vel_range = [95,105] #[64,128]
    pitch_range = [48,72]

    #####################
    grid_music_sigma = 10
    #####################

    for b in range(3):
        if b == 0:
            note_number = 8 *args.beat        # onset 개수
            midi_name = 'hihat_midi'          # 폴더,파일 이름
        elif b == 1:
            note_number = 2 *args.beat
            midi_name = 'kick_midi'
        elif b == 2 :
            note_number = 2 *args.beat
            midi_name = 'snare_midi'


        for _ in tqdm(range(midi_number)):
            # create an empty file
            mido_obj = mid_parser.MidiFile()
            beat_resol = mido_obj.ticks_per_beat

            # create an  instrument
            track = ct.Instrument(program=0, is_drum=True, name = midi_name)
            mido_obj.instruments = [track]

            # 1beat = 1920 / 1tick = 480
            midi_numpy = np.zeros([2,int(480 * tick_number)]) # midi_1920
            
            onsets = []
            for i in range(note_number):
                # create one note

                # onset(start)

                if args.grid_random == 'R':
                    start = np.random.randint(0, (tick_number * 480) -1) # whole random
                    if start not in onsets:
                        onsets.append(start)
                    else:
                        while start in onsets:
                            start = np.random.randint(0, (tick_number * 480) -1)
                        onsets.append(start)
                elif args.grid_random == 'G':
                    if b == 2: 
                        start = int(tick_number * 480 / note_number) * i + int(tick_number * 480 / note_number / 2)
                    else:
                        start = int(tick_number * 480 / note_number) * i 
                elif args.grid_random == 'RG':
                    start = np.random.randint(0, (tick_number * 4)) # random in grid
                    start *= 120
                    if start == 480*tick_number:
                        start -= 1
                    if start not in onsets:
                        onsets.append(start)
                    else:
                        while start in onsets:
                            start = np.random.randint(0, (tick_number * 16))
                            start *= 30
                            
                        onsets.append(start)
                elif args.grid_random == 'GG':
                    if i == 0 :
                        start = abs(int(tick_number * 480 / note_number) * i + int(np.random.normal(0, grid_music_sigma)))
                    else:
                        start = int(tick_number * 480 / note_number) * i + int(np.random.normal(0, grid_music_sigma)) 

                if start == 480*tick_number:
                    start -= 1
                end = start + 1

                #velocity & pitch
                if musig == True:
                    velocity = int(np.random.normal(velocity_mu, velocity_sigma))
                    pitch = 60
                    # pitch = int(np.random.normal(pitch_mu, pitch_sigma))
                else:
                    velocity = np.random.randint(vel_range[0],vel_range[1])
                    pitch = 60
                    # pitch = np.random.randint(pitch_range[0],pitch_range[1])

                note = ct.Note(start=start, end=end, pitch=pitch, velocity=velocity)

                mido_obj.instruments[0].notes.append(note)

                # start_n = int(start*(sec_per_tick * tick_number * sample_rate /(tick_number*480))) # start * 50 (1920 * 50  = 960000)
                midi_numpy[0,start] = (velocity+1)/128
                midi_numpy[1,start] = pitch

            # create markers
            marker_hi = ct.Marker(time=0, text='TT')
            mido_obj.markers.append(marker_hi)

            # write to file
            output_dir1 = output_dir + f'drum_data_{data_type}/generated_midi/{midi_name}'
            os.makedirs(output_dir1, exist_ok=True)
            output_file = os.path.join(output_dir1, f'{midi_name}_{_}.midi')
            mido_obj.dump(output_file)

            # write to midi file
            output_dir2 = output_dir + f'drum_data_{data_type}/generated_midi_numpy/{midi_name}_numpy'
            os.makedirs(output_dir2, exist_ok=True)
            np.save(output_dir2+f'/{midi_name}_{_}', midi_numpy)    


        print(f"generated '{midi_number}' number of '{midi_name}' succeed")
    return None

if __name__ == "__main__":
    # 기본적인 randomness
    # velocity 는 100 을 평균으로 gaussian distribution sigma = 3
    # pitch 는 60 고정 (C5)

    # python midi_generate.py --data_type train --midi_number 10 --bpm 120 --beat 1 --sample_rate 48000
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='prac', help='train, val, test')
    parser.add_argument('--midi_number', type=int, default=10, help='midi number')
    parser.add_argument('--beat', type=int, default=1, help='beat')
    parser.add_argument('--sample_rate', type=int, default=48000, help='sample_rate')
    parser.add_argument('--grid_random', type=str, default='RG', help='R for random, G for grid, RG for random in grid, GG for gaussian in grid')
    parser.add_argument('--random_type', type=str, default='random', help='random or gaussian')
    parser.add_argument('--output_dir', type=str, default='./generated_data/', help='data directory')
    args = parser.parse_args()
    generate_midi(args)