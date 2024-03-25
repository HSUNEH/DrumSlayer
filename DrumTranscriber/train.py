# from encoder_decoder_inst_c import EncoderDecoderModule, EncoderDecoderConfig    
from inst_decoder import InstDecoderModule, InstDecoderConfig
from dataset_c import DrumSlayerDataset


from torch.utils.data import DataLoader
import wandb
import lightning.pytorch as pl
import os
import datetime
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import argparse


# # Set the desired CUDA device number
# torch.cuda.set_device(7)
# device_number = torch.cuda.current_device()
# print(f"CUDA device number: {device_number}")


# class ValEveryNSteps(pl.Callback):
#     def __init__(self, every_n_steps, model, val_dataloaders):
#         super().__init__()
#         self.last_run = None
#         self.every_n_steps = every_n_steps
#         self.model = model
#         self.val_dataloaders = val_dataloaders

#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         if trainer.global_step == self.last_run:
#             return
#         else:
#             self.last_run = None
#         if trainer.global_step % self.every_n_steps == 0 and trainer.global_step != 0:
#             trainer.training = False
#             trainer.validating = True
#             trainer.validate(model=self.model,dataloaders=self.val_dataloaders)
#             trainer.validating = False
#             trainer.training = True
#             # trainer.logger_connector.epoch_end_reached = False
#             self.last_run = trainer.global_step


RESUME = False

audio_encoding_type = "codes" # "latents", "codes", "z" (dim: 72, 9, 1024)

# data_dir = '/workspace/DrumSlayer/generated_data/'
data_dir = '/disk2/st_drums/generated_data/'
trained_dir = '/workspace/DrumTranscriber/ckpts'

def main(args):
    
    BATCH_SIZE = args.batch_size # Default : 4 
    EXP_NAME = f"{datetime.datetime.now().strftime('%m-%d-%H-%M')}-STDT-{args.train_type}-{args.layer_cut}_{args.dim_cut}_{args.batch_size}"
    os.makedirs(f"{trained_dir}/{EXP_NAME}/", exist_ok=True)

    if args.wandb == True: 
        wandb.init(project="DrumTranscriber", name=EXP_NAME)
        WANDB = True
    else:
        WANDB = False    

    # debug_dataset = DrumSlayerDataset(data_dir, "debug", audio_encoding_type, args)
    # debug_dataloader = DataLoader(debug_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    train_dataset = DrumSlayerDataset(data_dir, "train", audio_encoding_type, args)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_dataset = DrumSlayerDataset(data_dir, "valid", audio_encoding_type, args)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # config = EncoderDecoderConfig(audio_rep = audio_encoding_type, args = args)
    # model = EncoderDecoderModule(config)
    
    config = InstDecoderConfig(audio_rep = audio_encoding_type, args = args)
    model = InstDecoderModule(config)


    # # LOAD PRETRAINED MODEL
    # ckpt_dir = '/workspace/DrumTranscriber/ckpts/03-24-14-09-STDT-kick-2_2_4/train_total_loss=2.15-valid_total_loss=11.76.ckpt'
    # ckpt = torch.load(ckpt_dir, map_location='cpu')
    # model.load_state_dict(ckpt['state_dict'])
    
    
    ddp_strategy = pl.strategies.DDPStrategy(find_unused_parameters=True)

    valid_n_steps = 2000
    check_point_n_steps = 2000
    
    n_step_checkpoint = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        monitor="train_total_loss",
        mode="min",
        dirpath=f"{trained_dir}/{EXP_NAME}/",

        filename = "{train_total_loss:.2f}-{valid_total_loss:.2f}", 
        # every_n_epochs = 1
        every_n_train_steps=check_point_n_steps, # n_steps
    )
    
    n_step_earlystop = pl.callbacks.EarlyStopping(                                                                                                                                                                    
                        monitor="valid_total_loss",                                                                                                                                                                        
                        min_delta=0.00,                                                                                                                                                                            
                        patience=5,                                                                                                                                                                                
                        verbose=True,                                                                                                                                                                              
                        mode="min",                                                                                                                                                                                
                        check_on_train_epoch_end=False,                                                                                                                                                            
                    )                                                                                                                                                                                          

    if WANDB:
        logger = WandbLogger(name=EXP_NAME, project="DrumSlayer")
        # trainer = pl.Trainer(accelerator="gpu", logger=logger, devices=NUM_DEVICES, max_epochs=5, precision='16-mixed', callbacks=[n_step_checkpoint, n_step_earlystop], strategy=ddp_strategy, val_check_interval=valid_n_steps)
        trainer = pl.Trainer(accelerator="gpu", logger=logger, devices=NUM_DEVICES, max_epochs=3, precision='16-mixed', callbacks=[n_step_checkpoint], strategy=ddp_strategy, val_check_interval=valid_n_steps)
    else:
        # logger = TensorBoardLogger(save_dir=f"{trained_dir}/{EXP_NAME}/logs", name=EXP_NAME)
        # trainer = pl.Trainer(accelerator="gpu", logger=logger, devices=NUM_DEVICES, max_epochs=5, precision='16-mixed', callbacks=[n_step_checkpoint, n_step_earlystop], strategy=ddp_strategy, val_check_interval=valid_n_steps)
        trainer = pl.Trainer(accelerator="gpu", devices=NUM_DEVICES, max_epochs=4, precision='16-mixed',  callbacks=[n_step_checkpoint], strategy=ddp_strategy)


    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_type', type=str, default='kick', help='ksh, kshm, kick, snare, hihat')
    parser.add_argument('--wandb', type=bool, default=False, help='True, False')
    parser.add_argument('--layer_cut', type=int, default='1', help='enc(or dec)_num_layers // layer_cut')
    parser.add_argument('--dim_cut', type=int, default='1', help='enc(or dec)_num_heads, _d_model // dim_cut')
    parser.add_argument('--batch_size', type=int, default='16', help='batch size')
    args = parser.parse_args()
    
    NUM_DEVICES = [3]
    
    NUM_WORKERS = 15

    main(args)
