from encoder_decoder_inst import EncoderDecoderModule, EncoderDecoderConfig    
from dataset import DrumSlayerDataset
from torch.utils.data import DataLoader
import wandb
import lightning.pytorch as pl
import os
import datetime
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import argparse

NUM_DEVICES = 4,5,6,7
NUM_WORKERS = 0

# # Set the desired CUDA device number
# torch.cuda.set_device(7)
# device_number = torch.cuda.current_device()
# print(f"CUDA device number: {device_number}")



RESUME = False

audio_encoding_type = "codes" # "latents", "codes", "z" (dim: 72, 9, 1024)

# data_dir = '/workspace/DrumSlayer/generated_data/'
data_dir = '/disk2/st_drums/generated_data/'
trained_dir = '/workspace/DrumTranscriber/ckpts'

def main(args):
    
    BATCH_SIZE = args.batch_size # Default : 4 
    EXP_NAME = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}-STDT-{args.train_type}-{args.layer_cut}_{args.dim_cut}_{args.batch_size}"
    os.makedirs(f"{trained_dir}/{EXP_NAME}/", exist_ok=True)

    if args.wandb == True: 
        wandb.init(project="DrumTranscriber", name=EXP_NAME)
        WANDB = True
    else:
        WANDB = False    

    train_dataset = DrumSlayerDataset(data_dir, "train", audio_encoding_type, args)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_dataset = DrumSlayerDataset(data_dir, "valid", audio_encoding_type, args)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    config = EncoderDecoderConfig(audio_rep = audio_encoding_type, args = args)

    model = EncoderDecoderModule(config)
    ddp_strategy = pl.strategies.DDPStrategy(find_unused_parameters=True)

    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     dirpath=f"{trained_dir}/{EXP_NAME}/",
    #     monitor="val_content_loss",
    #     mode = "min",
    #     every_n_epochs=1,
    #     filename = "{epoch}-{val_content_loss:.2f}",
    #     verbose=True,
    #     save_top_k=3,
    #     save_last=False,
    # )
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     dirpath=f"{trained_dir}/{EXP_NAME}/",
    #     # dirpath=f"/data5/kyungsu/ckpts/DrumSlayer/{EXP_NAME}/",
    #     monitor="valid_total_loss",
    #     mode = "min",
    #     # every_n_steps=100,
    #     filename = "{epoch}-{valid_total_loss:.2f}",
    #     verbose=True,
    #     save_top_k=3,
    #     save_last=False,
    # )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=10,
        monitor="train_total_loss",
        mode="min",
        dirpath=f"{trained_dir}/{EXP_NAME}/",
        filename = "{epoch}-{train_total_loss:.2f}",
        every_n_train_steps=5000, # n_steps
    )
    if WANDB:
        logger = WandbLogger(name=EXP_NAME, project="DrumSlayer")
        trainer = pl.Trainer(accelerator="gpu", logger=logger, devices=NUM_DEVICES, max_epochs=1, precision='16-mixed', callbacks=[checkpoint_callback], strategy=ddp_strategy)
    else:
        logger = TensorBoardLogger(save_dir=f"{trained_dir}/{EXP_NAME}/logs", name=EXP_NAME)
        trainer = pl.Trainer(accelerator="gpu", logger=logger, devices=NUM_DEVICES, max_epochs=1, precision='16-mixed', callbacks=[checkpoint_callback], strategy=ddp_strategy)


    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


    # config = DelayDecoderConfig(positional_encoding=POSITIONAL_ENCODING, num_projection_layers=2, wandb=WANDB)
    # if RESUME:
    #     model = DelayDecoderModule.load_from_checkpoint(f"/data5/kyungsu/ckpts/MALL-E/{EXP_NAME}/last.ckpt", config=config)
    # else:
    #     model = DelayDecoderModule(config)

    # if WANDB:
    #     wandb.config.update({"batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS})
    #     wandb.watch(model)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     dirpath=f"/data5/kyungsu/ckpts/MALL-E/{EXP_NAME}",
    #     filename = "last",
    #     verbose=True,
    #     every_n_train_steps=1000,
    #     save_top_k=-1,
    #     save_last=False,
    #     enable_version_counter=False
    # )
    
    # trainer = pl.Trainer(accelerator="gpu", devices=NUM_DEVICES, max_epochs=1, callbacks=[checkpoint_callback], precision='16-mixed')
    # trainer.fit(model=model, train_dataloaders=train_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_type', type=str, default='kick', help='ksh, kshm, kick, snare, hihat')
    parser.add_argument('--wandb', type=bool, default=True, help='True, False')
    parser.add_argument('--layer_cut', type=int, default='2', help='enc(or dec)_num_layers // layer_cut')
    parser.add_argument('--dim_cut', type=int, default='2', help='enc(or dec)_num_heads, _d_model // dim_cut')
    parser.add_argument('--batch_size', type=int, default='4', help='batch size')
    args = parser.parse_args()

    main(args)
