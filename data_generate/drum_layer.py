from pydub import AudioSegment
import os
import random
from tqdm import tqdm
from itertools import permutations
random.seed(0)
# # 오디오 파일 로드
# audio1 = AudioSegment.from_wav("0_kick_t.wav")
# audio2 = AudioSegment.from_wav("1_kick_t.wav")
total_number  = 1000000
for type in ['train', 'valid', 'test'] :
    for inst in ['kick', 'snare', 'hhclosed']:
            

        one_shots_dir = f'/disk2/st_drums/one_shots/{type}/{inst}/'
# one_shots_dir = f'/disk2/st_drums/one_shots/debug/kick/'
        wav_files = [os.path.join(one_shots_dir, f) for f in os.listdir(one_shots_dir) if f.endswith('.wav')]
        perms = list(permutations(wav_files, 2))
        random.shuffle(perms)

        n = 0
        if type == 'train':
            number = int(total_number*0.9)
        else:
            number = int(total_number*0.05)
        for i in tqdm(range(number), desc = f'{type} {inst}'):

            # 랜덤으로 두 개의 파일 선택
            
            while True:
                selected_files = perms[n]
                n +=1
                try : 
                    audio1 = AudioSegment.from_wav(selected_files[0])
                    audio2 = AudioSegment.from_wav(selected_files[1])
                    
                    break
                except: 
                    continue

            # 두 번째 오디오의 볼륨을 0.6, 0.5, 0.4배로 조절
            random_volume_down = random.choice([10, 15, 20])

            audio2 = audio2 - random_volume_down


            # 두 오디오 파일을 합치고 비율을 조절하여 합침
            combined_audio = audio1.overlay(audio2, position=0)


            # 합쳐진 오디오를 파일로 저장
            combined_audio.export(one_shots_dir +f"{i}_layered.wav", format="wav")
