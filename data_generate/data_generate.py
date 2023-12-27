import midi_2_wav
import argparse
import DAFXChain.drum_fx

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='all', help='all, train, valid, test')
    parser.add_argument('--other_sounds', type=bool, default=False, help='other sounds')
    parser.add_argument('--midi_number', type=int, default=10, help='midi number')
    # parser.add_argument('--beat', type=int, default=1, help='beat')
    parser.add_argument('--loop_seconds', type=int, default=5, help='loop_seconds')
    parser.add_argument('--sample_rate', type=int, default=44100, help='sample_rate')
    parser.add_argument('--grid_random', type=str, default='RG', help='R for random, G for grid, RG for random in grid, GG for gaussian in grid')
    parser.add_argument('--mono', type=bool, default=False, help='mono or stereo')
    parser.add_argument('--oneshot_dir', type=str, default='./midi_2_wav/one_shots/', help='input data directory')
    parser.add_argument('--output_dir', type=str, default='./generated_data/', help='output data directory')
    args = parser.parse_args()

    # TODO 1 : midi gen 
    midi_2_wav.generate_midi(args)

    # TODO 2 : midi 2 wav 
    midi_2_wav.generate_midi_2_wav(args)

    # TODO 3 : DAFX
    if args.other_sounds:
        DAFXChain.generate_drum_other_fx(args)
    else:
        DAFXChain.generate_drum_fx(args)

    # midi 10개, beat 1, sample rate 48000 -> 27m
    # -> 1개당 2.7m
    # 1000개 -> 2700m -> 약 2M
    # 1000000개 -> 약 2000M -> 약 2G
    # python data_generate.py --data_type all --midi_number 10
    # python data_generate.py --data_type all --oneshot_dir <데이터 저장공간> --output_dir <데이터 저장공간> --midi_number 1000000