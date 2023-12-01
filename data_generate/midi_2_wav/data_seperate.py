import os
import shutil
drums = ['kick', 'snare', 'hhclosed']

for drum in drums:

    # 파일 경로
    file_path = f'one_shots/dh_drumsets/{drum}'

    # 비율 설정
    train_ratio = 0.9
    valid_ratio = 0.05
    test_ratio = 0.05

    # 폴더 경로
    train_folder = f'one_shots/train/{drum}'
    valid_folder = f'one_shots/valid/{drum}'
    test_folder = f'one_shots/test/{drum}'

    # 파일 이동 함수
    def move_file(source, destination):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.move(source, destination)

    # 파일 이동
    files = os.listdir(file_path)
    total_files = len(files)
    train_files = int(total_files * train_ratio)
    valid_files = int(total_files * valid_ratio)
    test_files = total_files - train_files - valid_files

    for i, file in enumerate(files):
        source = os.path.join(file_path, file)
        if i < train_files:
            destination = os.path.join(train_folder, file)
        elif i < train_files + valid_files:
            destination = os.path.join(valid_folder, file)
        else:
            destination = os.path.join(test_folder, file)
        move_file(source, destination)
