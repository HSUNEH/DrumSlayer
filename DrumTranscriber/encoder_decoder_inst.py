import math
import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl
import pl_bolts
from tqdm import tqdm

from einops import rearrange


class EncoderDecoderConfig:
    def __init__(self, audio_rep, args, **kwargs):
        # Encoder hparams.
        self.enc_num_layers = 10//2
        self.enc_num_heads = 12//2
        self.enc_d_model = 768//2
        self.enc_d_ff = 3072
        # Decoder hparams.
        self.dec_num_layers = 10//2
        self.dec_num_heads = 12//2
        self.dec_d_model = 768//2
        self.dec_d_ff = 3072
        # Dropout hparams.
        self.embed_dropout = 0.1
        self.attn_dropout = 0.1
        self.ff_dropout = 0.1
        # Other hparams.
        self.midi_vocab_size = 1000+128+4+3 ## <start>,<end>,<pad>,<sep> 1000 time bins, 128 velocity bins, 3 drum types. 1134
        self.audio_vocab_size = 1024+4+1 # 1024 dac tokens + <start>,<end>,<pad>,<sep>
        # self.dac_vocab_size = 4+1000+128+1025 # 2157 # <start>,<end>,<pad>,<sep> 1000 time bins, 128 velocity bins, 3 drum types. + 1025 dac tokens
        # self.audio_rep = "latents" # "latents", "codes", "z"
        self.audio_rep = audio_rep
        if self.audio_rep == "latents":
            self.audio_rep_dim = 72
        elif self.audio_rep == "z":
            self.audio_rep_dim = 1024
        elif self.audio_rep == "codes":
            self.audio_rep_dim = 9
        else:
            raise ValueError("Invalid audio_rep type")
        self.max_target_len = 1472
        self.padding_loss_lambda = 0.1
        self.train_type = args.train_type
        if self.train_type not in ['ksh', 'kick', 'snare', 'hihat', 'kshm'] :
            raise ValueError("Invalid train_type")
        

class EncoderDecoderModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_decoder = EncoderDecoder(config)
        # self.loss_function = self.loss_functions = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim+1)] ##number
        if self.config.train_type == 'kshm':
            self.loss_functions = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim+1)] ##number
        else: 
            self.loss_functions = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim)] ##number
    
    def training_step(self, batch, batch_idx):
        total_loss, midi_loss, audio_losses = self.step(batch)
        self.log("train_total_loss", total_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        if self.config.train_type == 'kshm':
            self.log("train_midi_loss", midi_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_audio_loss", audio_losses.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        total_loss, midi_loss, audio_losses = self.step(batch)
        self.log("valid_total_loss", total_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        if self.config.train_type == 'kshm':
            self.log("valid_midi_loss", midi_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("valid_audio_loss", audio_losses.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return total_loss
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return self.optimizer
    
    def step(self, batch):
        x, y = batch # x: audio_rep, y: midi_audio_tokens  # x.shape : (4, TODO(431) , 2, 9), y.shape : (4, 10, 1472) 
        y = rearrange(y, 'b v t -> b t v') # (4, 1472, 10)
        y_pred = self.encoder_decoder(x, y[:,:-1]) # (batch_size, seq_len, total_vocab_size) torch.Size([4, 1471, 10396])
        
        if self.config.train_type == 'kshm': #kick snare hihat dac +  midi token
            # MIDI LOSS
            midi_target = rearrange(y[:,1:,0], 'b t -> (b t)') #5884 = b*1471
            dac_pred = rearrange(y_pred, 'b t v -> (b t) v') 

            midi_pred = rearrange(y_pred[:,:,:self.config.midi_vocab_size], 'b s v -> (b s) v')
            midi_loss = self.loss_functions[0](midi_pred, midi_target.long()) # CrossEntropyLoss

            
            # Audio DAC Loss
            audio_losses = []
            for i in range(9): ##number
                # y_pred : torch.Size([4, 2737, 2157]) b, t, v
                # y : torch.Size([4, 2738, 19])
                audio_logits = rearrange(y_pred[:,:,self.config.midi_vocab_size+i*self.config.audio_vocab_size:self.config.midi_vocab_size+(i+1)*self.config.audio_vocab_size], 'b s v -> (b s) v') # (4*2737, v)
                
                audio_target = rearrange(y[:,1:,i+1], 'b s -> (b s)').long() # 4*2738
                audio_losses.append(self.loss_functions[i+1](audio_logits, audio_target))
            total_loss = midi_loss + sum(audio_losses)
            loss = total_loss / 10.0 ##number

            return loss, midi_loss, sum(audio_losses)/9.0 ##number
        
        elif self.config.train_type == 'ksh':
            # MIDI LOSS
            midi_loss =0 
            
            # TODO : Audio DAC Loss
            audio_losses = []
            for i in range(9): ##number
                # y_pred : torch.Size([4, 2737, 2157]) b, t, v
                # y : torch.Size([4, 2738, 19])
                audio_logits = rearrange(y_pred[:,:,self.config.midi_vocab_size+i*self.config.audio_vocab_size:self.config.midi_vocab_size+(i+1)*self.config.audio_vocab_size], 'b s v -> (b s) v') # (4*2737, v)
                audio_target = rearrange(y[:,1:,i], 'b s -> (b s)').long() # 4*2738
                audio_losses.append(self.loss_functions[i](audio_logits, audio_target))
                
                # audio_target = rearrange(y[:,1:,i+1], 'b s -> (b s)').long() # 4*2738
                # audio_losses.append(self.loss_functions[i+1](audio_logits, audio_target))
            total_loss = midi_loss + sum(audio_losses)
            # loss = total_loss / 10.0 ##number
            loss = total_loss / 9 ##number
            
            return loss, midi_loss, sum(audio_losses)/9.0 ##number

        else: #inst
            midi_loss =0 
            
            # TODO : Audio DAC Loss
            audio_losses = []
            for i in range(9): ##number
                # y_pred : torch.Size([4, 2737, 2157]) b, t, v
                # y : torch.Size([4, 2738, 19])
                audio_logits = rearrange(y_pred[:,:,self.config.midi_vocab_size+i*self.config.audio_vocab_size:self.config.midi_vocab_size+(i+1)*self.config.audio_vocab_size], 'b s v -> (b s) v') # (4*2737, v)
                audio_target = rearrange(y[:,1:,i], 'b s -> (b s)').long() # 4*2738
                audio_losses.append(self.loss_functions[i](audio_logits, audio_target))
                
                # audio_target = rearrange(y[:,1:,i+1], 'b s -> (b s)').long() # 4*2738
                # audio_losses.append(self.loss_functions[i+1](audio_logits, audio_target))
            total_loss = midi_loss + sum(audio_losses)
            # loss = total_loss / 10.0 ##number
            loss = total_loss / 9 ##number
            
            return loss, midi_loss, sum(audio_losses)/9.0 ##number
            

class EncoderDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.audio_projection_layer = nn.Sequential(
            nn.Linear(self.config.audio_rep_dim, self.config.enc_d_model), # 9, 768
            nn.GELU(),
            nn.Linear(self.config.enc_d_model, self.config.enc_d_model) # 768, 768
        )
        self.audio_token_projection_layer = nn.Sequential(
            nn.Linear((self.config.audio_rep_dim)*2+1, self.config.enc_d_model), # 19, 768
            nn.GELU(),
            nn.Linear(self.config.enc_d_model, self.config.enc_d_model) # 768, 768
        )
        
        self.positional_encoder = PositionalEncoder(self.config.dec_d_model, self.config.embed_dropout) # 768, 0.1
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.softmax = nn.Softmax(dim=-1)
        self.dac_projection_layer = nn.Linear(config.dec_d_model, config.midi_vocab_size + config.audio_vocab_size * 9) ##number
        self.embedding = nn.Embedding(1025, 1)
        self.midi_embedding_layer = nn.Embedding(config.midi_vocab_size, config.dec_d_model)
        self.audio_embedding_layer = nn.ModuleList([nn.Embedding(config.audio_vocab_size, config.dec_d_model) for _ in range(9)]) ##number

    def forward(self, x, y=None, strategy="greedy", sample_arg=None):
        # Input
        # x: [batch_size, seq_len, 2, audio_rep_dim:9] 2 for stereo.        torch.Size([4, TODO(431), 2, 9])  audio_rep
        # y: [batch_size, seq_len, 10]                                      torch.Size([4, 1471, 10])   target_midi_audio_token
        # Output
        
        x = x.long()
        if self.config.audio_rep == "codes":
            x = self.embedding(x).squeeze(-1) # [batch_size, seq_len, 2, dac_dim(9)] -> [batch_size, seq_len, 2, dac_dim(9)]
        x_emb = self.audio_projection_layer(x) # [batch_size, seq_len, 2, dac_dim(9)] -> [batch_size, seq_len, 2, d_model(768)]
        x_emb_l = self.positional_encoder(x_emb[:,:,0]) # left : [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        x_emb_r = self.positional_encoder(x_emb[:,:,1]) # right : [batch_size, seq_len, d_model]
        x_emb = torch.cat([x_emb_l, x_emb_r], dim=1) # [batch_size, 2*seq_len, d_model], Concatenate stereo channels. 
        enc_output = self.encoder(x_emb) # [batch_size, seq_len, d_model]:  [4, TODO(2*431), 768] #TODO : concat 이 맘에 안들면 Parallel로 바꿔보기
        

        if self.config.train_type == 'kshm':
            tok_embedding = self.midi_embedding_layer(y[:,:,0]) # torch.Size([4, 1471]) -> torch.Size([4, 1471, 768])
        else:
            tok_embedding = None
            
        for i in range(9): ##number
            if self.config.train_type == 'kshm':
                audio_tok_embedding = self.audio_embedding_layer[i](y[:,:,i+1])
                audio_tok_embedding = self.midi_embedding_layer(y[:,:,i+1])
            else: 
                audio_tok_embedding = self.midi_embedding_layer(y[:,:,i])
                if tok_embedding is None:
                    tok_embedding = audio_tok_embedding
                tok_embedding += audio_tok_embedding # torch.Size([4, 1471, 768]) #TODO: 덧셈이 맞는지 확인 -> 맞다고 하네요. but 다른 방법론도 적용해보자

        tok_embedding = self.positional_encoder(tok_embedding) # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model] :[4, 1471, 768]
        y_out = self.decoder(tok_embedding, enc_output) # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        audio_logits = self.dac_projection_layer(y_out) # [batch_size, seq_len, config.midi_vocab_size + config.audio_vocab_size * 9]
        return audio_logits # y_pred : torch.Size([4, 1471, 10394])


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer("pe", self.get_positional_encoding())
    
    def get_positional_encoding(self):
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x):
        # Input
        # x: [batch_size, seq_len, d_model]
        # Output
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.enc_num_layers)])
    
    def forward(self, x):
        # Input
        # x: [batch_size, seq_len, d_model]
        # Output
        # x: [batch_size, seq_len, d_model]
        for layer in self.layers:
            x = layer(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.enc_d_model
        ff_dim = config.enc_d_ff
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=config.enc_num_heads, dropout=config.attn_dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(config.ff_dropout),
        )
    
    def forward(self, x):
        x = self.ln1(x)
        x = x + self.mha(x, x, x)[0]
        x = x + self.ff(self.ln2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.dec_num_layers)])
    
    def forward(self, x, enc_output):
        # Input
        # x: [batch_size, seq_len, d_model]
        # enc_output: [batch_size, seq_len, d_model]
        # Output
        # x: [batch_size, seq_len, d_model]
        for layer in self.layers:
            x = layer(x, enc_output)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.dec_d_model
        ff_dim = config.dec_d_ff
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.mha1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=config.dec_num_heads, dropout=config.attn_dropout, batch_first=True)
        self.mha2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=config.dec_num_heads, dropout=config.attn_dropout, batch_first=True)
        self.register_buffer("causal_mask", torch.triu(torch.ones(config.max_target_len, config.max_target_len).bool(), diagonal=1))
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(config.ff_dropout),
        )
    
    def forward(self, x, enc_output):
        x = self.ln1(x)
        x = x + self.mha1(x, x, x, is_causal=True, attn_mask=self.causal_mask[:x.size(1),:x.size(1)])[0]
        x = x + self.mha2(self.ln2(x), enc_output, enc_output, is_causal=False)[0]
        x = x + self.ff(self.ln3(x))
        return x    




if __name__ == "__main__":
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
    from torch.utils.data import Dataset
    import numpy as np
    import pretty_midi
    import glob

    
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
    audio_encoding_type = "latents" # "latents", "codes", "z" (dim: 72, 9, 1024)
    trained_dir = '/workspace/DrumTranscriber/ckpts'
    # data_dir = '/workspace/DrumSlayer/generated_data/'
    # data_dir = '/disk2/st_drums/generated_data/'
    # trained_dir = '/workspace/DrumTranscriber/ckpts'
    class TestDataset(Dataset):
        def __init__(self, max_len=1472, audio_encoding_type = "codes"):
            assert audio_encoding_type in ["latents", "codes", "z"] # dim: 72, 9, 1024
            self.file_path =  'disk2/st_drums/generated_data'

            self.max_len = max_len
            self.encoding_type = audio_encoding_type
            if self.encoding_type == 'codes':
                self.size = [10, max_len] ##number
        def __getitem__(self, idx):
            # audio_rep = np.load(self.file_path + f"drum_data_{self.split}/mixed_loops/{idx}_{self.encoding_type}.npy") ## npy 생성 -> preprocess_dac.py
            # audio_rep = rearrange(audio_rep, 'c d t -> t c d') # c: channel, d: dim, t: time
            audio_rep = np.random.randint(0, 1024, (2, 9, 431))
            audio_rep = rearrange(audio_rep, 'c d t -> t c d')
            
            # kick_midi =  pretty_midi.PrettyMIDI(self.file_path + f"drum_data_{self.split}/generated_midi/kick_midi/kick_midi_{idx}.midi")
            # snare_midi =  pretty_midi.PrettyMIDI(self.file_path + f"drum_data_{self.split}/generated_midi/snare_midi/snare_midi_{idx}.midi")
            # hihat_midi = pretty_midi.PrettyMIDI(self.file_path + f"drum_data_{self.split}/generated_midi/hihat_midi/hihat_midi_{idx}.midi")
            # midi_tokens = self.tokenize_midi(kick_midi, snare_midi, hihat_midi) # (,152)
            
            audio_tokens = np.random.randint(0,1024, (self.size)) # (d:10, t:2737) # TODO : stereo d => 19
            audio_tokens = rearrange(audio_tokens, 'd t -> t d') # (2737, 10)
            return audio_rep, audio_tokens # midi_tokens

        def __len__(self):
            return 1000

    def main():
        os.makedirs(f"{trained_dir}/{EXP_NAME}/", exist_ok=True)    
        # wandb.init(project="DrumTranscriber", name=EXP_NAME)
        max_len = 1472 
        audio_encoding_type = 'codes'
        train_path = 'disk2/st_drums/generated_data'
        train_dataset = TestDataset(max_len, audio_encoding_type)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        
        valid_dataset = TestDataset(max_len,  audio_encoding_type)
        valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        config = EncoderDecoderConfig(audio_rep = audio_encoding_type)

        model = EncoderDecoderModule(config)
        
        # checkpoint_callback = pl.callbacks.ModelCheckpoint(
        #     dirpath=f"{trained_dir}/{EXP_NAME}/",
        #     # dirpath=f"/data5/kyungsu/ckpts/DrumSlayer/{EXP_NAME}/",
        #     monitor="train_content_loss",
        #     mode = "min",
        #     every_n_steps=100,
        #     filename = "{epoch}-{train_content_loss:.2f}",
        #     verbose=True,
        #     save_top_k=3,
        #     save_last=False,
        # )
        if WANDB:
            logger = WandbLogger(name=EXP_NAME, project="DrumSlayer")
            trainer = pl.Trainer(accelerator="gpu", logger=logger, devices=NUM_DEVICES, max_epochs=1, precision='16-mixed', )
        else:
            logger = TensorBoardLogger(save_dir=f"{trained_dir}/{EXP_NAME}/logs", name=EXP_NAME)
            trainer = pl.Trainer(accelerator="gpu", logger=logger, devices=NUM_DEVICES, max_epochs=1, precision='16-mixed',)
        
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    main()