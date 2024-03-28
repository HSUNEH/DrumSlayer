import os

for type in ['train', 'valid', 'test']:
    for inst in ['kick', 'snare', 'hhclosed']:
        directory = f'/disk2/st_drums/one_shots/{type}/{inst}'
        for filename in os.listdir(directory):
            if filename.endswith('_layered.wav'):
                file_path = os.path.join(directory, filename)
                os.remove(file_path)
