import math

import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl
import wandb
import pl_bolts

from einops import rearrange

class DelayDecoderConfig:
    def __init__(self, positional_encoding=False, num_projection_layers=3, wandb=False, transcription_mode=False, midi_vocab_size=1000+128+4+1):
        self.num_heads = 16
        self.num_blocks = 12
        self.embed_dim = 1024
        self.attn_dropout = 0.1
        self.embed_dropout = 0.1
        self.ff_dropout = 0.1
        self.midi_vocab_size = midi_vocab_size
        self.audio_vocab_size = 1024 + 4
        self.max_len = 1200
        self.positional_encoding = positional_encoding
        assert num_projection_layers in [1, 2, 3]
        self.num_projection_layers = num_projection_layers
        self.wandb = wandb
        self.transcription_mode = transcription_mode

class DelayDecoderModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.midi_vocab_size = config.midi_vocab_size
        self.audio_vocab_size = config.audio_vocab_size
        self.decoder = DelayDecoder(config)
        self.loss_functions = [nn.CrossEntropyLoss() for _ in range(10)]
        self.save_hyperparameters()
    
    def training_step(self, batch, batch_idx):
        clap_embedding, token = batch
        token = token.long()
        x = token[:, :-1]
        target = token 
        logits = self.decoder(clap_embedding, x) # logits.shape == (batch_size, seq_len, total_vocab_size)


        # Calculate Loss.
        midi_logits = rearrange(logits[:,:,:self.midi_vocab_size], 'b s v -> (b s) v')
        midi_target = rearrange(target[:,:,0], 'b s -> (b s)')
        midi_loss = self.loss_functions[0](midi_logits, midi_target)
        audio_losses = []
        for i in range(9):
            audio_logits = rearrange(logits[:,:,self.midi_vocab_size+i*self.audio_vocab_size:self.midi_vocab_size+(i+1)*self.audio_vocab_size], 'b s v -> (b s) v')
            audio_target = rearrange(target[:,:,i+1], 'b s -> (b s)')
            audio_losses.append(self.loss_functions[i+1](audio_logits, audio_target))
        total_loss = midi_loss + sum(audio_losses)
        loss = total_loss / 10.0

        # Calculate Accuracy.
        midi_pred = torch.argmax(midi_logits, dim=-1)
        midi_acc = torch.sum(midi_pred == midi_target).float() / torch.numel(midi_target)
        audio_accs = []
        for i in range(9):
            audio_logits = rearrange(logits[:,:,self.midi_vocab_size+i*self.audio_vocab_size:self.midi_vocab_size+(i+1)*self.audio_vocab_size], 'b s v -> (b s) v')
            audio_target = rearrange(target[:,:,i+1], 'b s -> (b s)')
            audio_pred = torch.argmax(audio_logits, dim=-1)
            audio_acc = torch.sum(audio_pred == audio_target).float() / torch.numel(audio_target)
            audio_accs.append(audio_acc)
        mean_acc = (midi_acc + sum(audio_accs)) / 10.0 
        
        if self.config.wandb:
            wandb.log({"train_loss": loss.item(), "train_midi_loss": midi_loss.item(), "train_audio_loss": sum(audio_losses)/len(audio_losses),
                    "train_midi_acc": midi_acc.item(), "train_audio_acc": sum(audio_accs)/len(audio_accs), "train_mean_acc": mean_acc.item()})
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        clap_embedding, token = batch
        token = token.long()
        x = token[:, :-1]
        target = token 
        logits = self.decoder(clap_embedding, x)
        
        # Calculate Loss.
        midi_logits = rearrange(logits[:,:,:self.midi_vocab_size], 'b s v -> (b s) v')
        midi_target = rearrange(target[:,:,0], 'b s -> (b s)')
        midi_loss = self.loss_functions[0](midi_logits, midi_target)
        audio_losses = []
        for i in range(9):
            audio_logits = rearrange(logits[:,:,self.midi_vocab_size+i*self.audio_vocab_size:self.midi_vocab_size+(i+1)*self.audio_vocab_size], 'b s v -> (b s) v')
            audio_target = rearrange(target[:,:,i+1], 'b s -> (b s)')
            audio_losses.append(self.loss_functions[i+1](audio_logits, audio_target))
        total_loss = midi_loss + sum(audio_losses)
        loss = total_loss / 10.0

        # Calculate Accuracy.
        midi_pred = torch.argmax(midi_logits, dim=-1)
        midi_acc = torch.sum(midi_pred == midi_target).float() / torch.numel(midi_target)
        audio_accs = []
        for i in range(9):
            audio_logits = rearrange(logits[:,:,self.midi_vocab_size+i*self.audio_vocab_size:self.midi_vocab_size+(i+1)*self.audio_vocab_size], 'b s v -> (b s) v')
            audio_target = rearrange(target[:,:,i+1], 'b s -> (b s)')
            audio_pred = torch.argmax(audio_logits, dim=-1)
            audio_acc = torch.sum(audio_pred == audio_target).float() / torch.numel(audio_target)
            audio_accs.append(audio_acc)
        mean_acc = (midi_acc + sum(audio_accs)) / 10.0 

        if self.config.wandb:
            wandb.log({"val_loss": loss.item(), "val_midi_loss": midi_loss.item(), "val_audio_loss": sum(audio_losses)/len(audio_losses),
                    "val_midi_acc": midi_acc.item(), "val_audio_acc": sum(audio_accs)/len(audio_accs), "val_mean_acc": mean_acc.item()})

        return loss
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return self.optimizer

class DelayDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.embed_dim
        self.max_len = config.max_len
        
        if config.num_projection_layers == 1:
            self.clap_projection = nn.Sequential(nn.Linear(512, embed_dim, bias=True))
        elif config.num_projection_layers == 2:
            self.clap_projection = nn.Sequential(nn.Linear(512, 1024, bias=True), nn.ReLU(), nn.Linear(1024, embed_dim, bias=True))
        elif config.num_projection_layers == 3:
            self.clap_projection = nn.Sequential(nn.Linear(512, 1024, bias=True), nn.ReLU(), nn.Linear(1024, 1024, bias=True), nn.ReLU(), nn.Linear(1024, embed_dim, bias=True))
        else:
            raise ValueError("invalid number of projection layers")
        
        if self.config.transcription_mode:
            self.sos_tok_embed = nn.Embedding(1, embed_dim)    
        self.midi_tok_embed = nn.Embedding(config.midi_vocab_size, embed_dim)
        self.audio_tok_embed = nn.ModuleList([nn.Embedding(config.audio_vocab_size, embed_dim) for _ in range(9)])
        if config.positional_encoding:
            self.pos_embed = nn.Parameter(torch.zeros(1, config.max_len, embed_dim))
        self.dropout = nn.Dropout(config.embed_dropout)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.num_blocks)]
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, config.midi_vocab_size+9*config.audio_vocab_size)
    
    def forward(self, clap_embedding, x):
        """
        clap_embedding.shape = (batch_size, 512)
        x.shape = (batch_size, seq_len, 10)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)+1
        assert seq_len <= self.max_len, "sequence longer than model capacity"
        # If transcription mode, first token is sos token.
        if self.config.transcription_mode:
            first_embedding = self.sos_tok_embed(torch.zeros(batch_size, 1).long().cuda())
        # Else, first token is clap embedding.
        else:
            first_embedding = self.clap_projection(clap_embedding).unsqueeze(1) # -> (batch_size, 1, embed_dim)

        if x.size(1) == 0:
            embedding = first_embedding
        else:
            tok_embedding = self.midi_tok_embed(x[:,:,0])
            for i in range(9):
                audio_tok_embedding = self.audio_tok_embed[i](x[:,:,i+1])
                tok_embedding = tok_embedding + audio_tok_embedding # -> (batch_size, seq_len, embed_dim)
            embedding = torch.cat((first_embedding, tok_embedding), dim=1)

        if self.config.positional_encoding:
            embedding = embedding + self.pos_embed[:, :seq_len, :]

        # tok_embedding.shape == (batch_size, seq_len, embed_dim)
        x = self.dropout(embedding)
        x = self.blocks(x)
        x = self.ln(x)
        x = self.fc(x)
        # x.shape == (batch_size, seq_len, vocab_size)
        return x
     
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.embed_dim
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(config)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(config.ff_dropout),
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        assert embed_dim % self.num_heads == 0, "invalid heads and embedding dimension configuration"
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.proj_dropout = nn.Dropout(config.ff_dropout)
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.max_len, config.max_len).bool())
            .unsqueeze(0).unsqueeze(0)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        # x.shape == (batch_size, seq_len, embed_dim)
        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1,2)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape == (batch_size, num_heads, seq_len, head_dim)

        mask = self.mask[:, :, :seq_len, :seq_len]
        y = F.scaled_dot_product_attention(q, k_t, v, is_causal=True)
        
        y = y.transpose(1, 2)
        # y.shape == (batch_size, seq_len, num_heads, head_dim)
        y = y.reshape(batch_size, seq_len, -1)
        # y.shape == (batch_size, seq_len, embed_dim)
        y = self.proj_dropout(self.proj(y))
        return y
