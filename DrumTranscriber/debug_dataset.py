num_sample = 15



import shutil
import os
import shutil

source_folder = '/disk2/st_drums/generated_data/drum_data_train/mixed_loops/' 
destination_folder = '/disk2/st_drums/generated_data/drum_data_debug/mixed_loops/'
if os.path.exists(destination_folder):
    shutil.rmtree(destination_folder)

os.mkdir(destination_folder)


for i in range(num_sample):
    shutil.copy(source_folder +f'{i}.wav', destination_folder)
    shutil.copy(source_folder +f'{i}_codes.npy', destination_folder)

original_file = '/disk2/st_drums/generated_data/drum_data_train/kickShotList.txt'
destination_file = '/disk2/st_drums/generated_data/drum_data_debug/kickShotList.txt'

if os.path.exists(destination_file):
    os.remove(destination_file)
    
original_dir = '/disk2/st_drums/one_shots/train/kick/'
destination_dir = '/disk2/st_drums/one_shots/debug/kick'
if os.path.exists(destination_dir):
    shutil.rmtree(destination_dir)

os.mkdir(destination_dir)


with open(original_file, 'r') as source_file, open(destination_file, 'w') as dest_file:
    for i, line in enumerate(source_file):
        dest_file.write(line)

        shutil.copy(original_dir + line.strip(), destination_dir)
        shutil.copy(original_dir + line.strip()[:-4]+ '_codes.npy', destination_dir)
        if i == num_sample-1:
            break
