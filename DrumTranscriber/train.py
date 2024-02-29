from encoder_decoder_m import EncoderDecoderModule, EncoderDecoderConfig    
from dataset import DrumSlayerDataset
from torch.utils.data import DataLoader
import wandb
import lightning.pytorch as pl
import os
import datetime
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger
import torch

BATCH_SIZE = 2
NUM_WORKERS = 0
NUM_DEVICES = 0,1,2,3,4
# # Set the desired CUDA device number
# torch.cuda.set_device(7)
# device_number = torch.cuda.current_device()
# print(f"CUDA device number: {device_number}")

EXP_NAME = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}-lucky-seven"

RESUME = False
WANDB = False
audio_encoding_type = "codes" # "latents", "codes", "z" (dim: 72, 9, 1024)

# data_dir = '/workspace/DrumSlayer/generated_data/'
data_dir = '/disk2/st_drums/generated_data/'
trained_dir = '/workspace/DrumTranscriber/ckpts'

def main():
    os.makedirs(f"{trained_dir}/{EXP_NAME}/", exist_ok=True)
    # wandb.init(project="DrumTranscriber", name=EXP_NAME)
    
    train_dataset = DrumSlayerDataset(data_dir, "train", audio_encoding_type)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_dataset = DrumSlayerDataset(data_dir, "valid", audio_encoding_type)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    config = EncoderDecoderConfig(audio_rep = audio_encoding_type)

    model = EncoderDecoderModule(config)

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
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{trained_dir}/{EXP_NAME}/",
        # dirpath=f"/data5/kyungsu/ckpts/DrumSlayer/{EXP_NAME}/",
        monitor="train_content_loss",
        mode = "min",
        # every_n_steps=100,
        filename = "{epoch}-{train_content_loss:.2f}",
        verbose=True,
        save_top_k=3,
        save_last=False,
    )
    if WANDB:
        logger = WandbLogger(name=EXP_NAME, project="DrumSlayer")
        trainer = pl.Trainer(accelerator="gpu", logger=logger, devices=NUM_DEVICES, max_epochs=1, precision='16-mixed', callbacks=[checkpoint_callback])
    else:
        logger = TensorBoardLogger(save_dir=f"{trained_dir}/{EXP_NAME}/logs", name=EXP_NAME)
        trainer = pl.Trainer(accelerator="gpu", logger=logger, devices=NUM_DEVICES, max_epochs=1, precision='16-mixed', callbacks=[checkpoint_callback])


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
    main()
    