import os
import shutil
import re
# 원본 폴더 경로
source_folder = "/Users/hwang/Music/Logic/Splice/sounds/packs"

# 새로운 폴더 경로
drums = ['kick', 'snare', 'hihat']


for drum in drums:
    destination_folder = f"/Users/hwang/Music/Logic/Splice/sounds/{drum}s"

    # 기존 폴더 삭제 후 생성
    shutil.rmtree(destination_folder, ignore_errors=True)
    os.makedirs(destination_folder, exist_ok=True)

    count = 0
    # 원본 폴더 내의 모든 파일과 폴더에 대해 반복
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(".wav") and (f"{drum}" in file.lower()) and (f"{drum}s" not in file.lower() and "fill" not in file.lower() and "open" not in file.lower() and "roll" not in file.lower() and "bpm" not in file.lower() and "loop" not in file.lower()):
                # Exclude words containing three-digit numbers other than '808'
                if not any(char.isdigit() and len(char) == 3 and char != '808' for word in file.lower().split() for char in re.split(r'[_\.]', word)):


                    # 원본 파일 경로
                    source_file = os.path.join(root, file)
                    # 새로운 파일 경로
                    destination_file = os.path.join(destination_folder, file)
                    # 파일 복사
                    shutil.copy2(source_file, destination_file)
                    count += 1 

    print(f'{drum}에서', count, "개의 파일이 복사되었습니다.")


# import os
# import scipy.io.wavfile

# # 'hihats' 폴더 경로
# folder_path = '/Users/hwang/Music/Logic/Splice/sounds/hihats'

# # 폴더 내의 모든 파일에 대해 반복
# for filename in os.listdir(folder_path):
#     if filename.endswith('.wav'):  # wav 파일인 경우
#         file_path = os.path.join(folder_path, filename)
#         sample_rate, data = scipy.io.wavfile.read(file_path)
#         duration = len(data) / sample_rate
#         if duration > 5:  # 길이가 5초를 초과하는 경우
#             os.remove(file_path)  # 파일 삭제

