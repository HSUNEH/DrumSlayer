import numpy
import glob
from tqdm import tqdm
#import dataset and dataloader from torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse

import dac
from audiotools import AudioSignal


import audiofile as af
import torch

BATCH_SIZE = 16
NUM_WORKERS = 2

# # Set the desired CUDA device number
# torch.cuda.set_device(1)
# device_number = torch.cuda.current_device()
# print(f"CUDA device number: {device_number}")


def gpu_setting(device_ids):
    # Get the number of available CUDA devices
    device_count = torch.cuda.device_count()

    if device_count < 8:
        print("Not enough CUDA devices (need at least 8).")
        return

    # Specify the device IDs you want to use (4, 5, 6, 7)


    for device_id in device_ids:
        # Check if the specified device ID is valid
        if device_id < 0 or device_id >= device_count:
            print(f"Invalid device ID: {device_id}")
            continue

        # Set the current CUDA device
        torch.cuda.set_device(device_id)

        # Perform operations on the current CUDA device

        # Print device properties
        device_prop = torch.cuda.get_device_properties(device_id)
        print(f"Device {device_id}: {device_prop.name}")

        # Reset the device to the default state (optional)
        torch.cuda.empty_cache()

def collate_fn(batch):
    wav_list = []
    fname_list = []
    for wav, fname in batch:
        wav_list.append(wav)
        fname_list.append(fname)
    return torch.stack(wav_list), fname_list

class DrumLoopDataset(Dataset):
    def __init__(self, file_path, split):
        assert split in ["train", "valid", "test"]
        self.file_path = file_path
        self.fnames = sorted(glob.glob(file_path + f"/drum_data_{split}/mixed_loops/*.wav"))
        print(len(self.fnames))
    
    def __getitem__(self, idx):
        # Use the fastest loading and resampling method
        wav, sr = af.read(self.fnames[idx])
        return wav, self.fnames[idx]
    
    def __len__(self):
        return len(self.fnames)


def main():
    # path = '/workspace/DrumSlayer/generated_data/'
    path = '/disk2/st_drums/generated_data/'
    
    model_path = dac.utils.download(model_type="44khz")
    model = dac.DAC.load(model_path)    
    
    device_ids = [7]
    gpu_setting(device_ids)
    
    train_dataset = DrumLoopDataset(path, "train")
    valid_dataset = DrumLoopDataset(path, "valid")
    test_dataset = DrumLoopDataset(path, "test")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)


    model = model.cuda()

    for loader in [train_dataloader, valid_dataloader, test_dataloader]:
        for wav, fname in tqdm(loader):
            
            wav = wav.reshape([-1,1,220500]).cuda() # Merge batch and channel. (torch.Size [32, 1, 220500]-> [16, 2, 220500])
            with torch.no_grad(): # dim ["latents" 72, "codes" 9 , "z" 1024] 
                x = model.preprocess(wav, 44100)
                z, codes, latents,_ ,_ = model.encode(x)
                # z_np = z.cpu().numpy()
                codes_16bit = codes.cpu().numpy().astype('int16')
                # latents_np = latents.cpu().numpy()
                
            for i, fname in enumerate(fname):
                # numpy.save(fname.replace(".wav", "_z.npy"), z_np[2*i:2*(i+1)]) 
                numpy.save(fname.replace(".wav", "_codes.npy"), codes_16bit[2*i:2*(i+1)]) # shape : (2,9,431) -> stereo, 9 audio_rep_dim, 431 seq_len
                # numpy.save(fname.replace(".wav", "_latents.npy"), latents_np[2*i:2*(i+1)])



if __name__ == "__main__":
    main()
