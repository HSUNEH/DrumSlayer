import math
import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl
import pl_bolts
from tqdm import tqdm

from einops import rearrange

# TODO:
# Key padding mask for variable length src input.

class EncoderDecoderConfig:
    def __init__(self, audio_rep, **kwargs):
        # Encoder hparams.
        self.enc_num_layers = 10
        self.enc_num_heads = 12
        self.enc_d_model = 768
        self.enc_d_ff = 3072
        # Decoder hparams.
        self.dec_num_layers = 10
        self.dec_num_heads = 12
        self.dec_d_model = 768
        self.dec_d_ff = 3072
        # Dropout hparams.
        self.embed_dropout = 0.1
        self.attn_dropout = 0.1 
        self.ff_dropout = 0.1
        # Other hparams.
        self.midi_vocab_size = 3+1000+128+3 # <start>,<end>,<pad> 1000 time bins, 128 velocity bins, 3 drum types. 
        # self.audio_rep = "latents" # "latents", "codes", "z
        self.audio_rep = audio_rep
        if self.audio_rep == "latents":
            self.audio_rep_dim = 72
        elif self.audio_rep == "z":
            self.audio_rep_dim = 1024
        else:
            self.audio_rep_dim = 9
        self.max_target_len = 151
        self.padding_loss_lambda = 0.1

class EncoderDecoderModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_decoder = EncoderDecoder(config)
        self.loss_function = nn.CrossEntropyLoss(reduction="none")
    
    def training_step(self, batch, batch_idx):
        padding_loss, content_loss, total_loss = self.step(batch)
        self.log("train_total_loss", total_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_padding_loss", padding_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_content_loss", content_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        padding_loss, content_loss, total_loss = self.step(batch)
        self.log("val_total_loss", total_loss.item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_padding_loss", padding_loss.item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_content_loss", content_loss.item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return total_loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return self.optimizer
    
    def step(self, batch):
        x, y = batch # x: audio_rep, y: midi_tokens
        y_pred = self.encoder_decoder(x, y[:,:-1])
        y_pred = rearrange(y_pred, 'b t v -> (b t) v')
        y_target = rearrange(y[:,1:], 'b t -> (b t)')

        non_reduced_loss = self.loss_function(y_pred, y_target.long())
        loss_for_padding = non_reduced_loss * (y_target == 2).float()
        loss_for_padding = loss_for_padding.sum() / (y_target == 2).float().sum()
        loss_for_content = non_reduced_loss * (y_target != 2).float()
        loss_for_content = loss_for_content.sum() / (y_target != 2).float().sum()

        total_loss = loss_for_content + self.config.padding_loss_lambda * loss_for_padding
        return loss_for_padding, loss_for_content, total_loss

class EncoderDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.audio_projection_layer = nn.Sequential(
            nn.Linear(self.config.audio_rep_dim, self.config.enc_d_model),
            nn.GELU(),
            nn.Linear(self.config.enc_d_model, self.config.enc_d_model)
        )
        self.midi_embedding_layer = nn.Embedding(self.config.midi_vocab_size, self.config.dec_d_model)
        self.positional_encoder = PositionalEncoder(self.config.dec_d_model, self.config.embed_dropout)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.softmax = nn.Softmax(dim=-1)
        self.midi_projection_layer = nn.Linear(config.dec_d_model, config.midi_vocab_size)
    
    def forward(self, x, y=None, strategy="greedy", sample_arg=None):
        # Input
        # x: [batch_size, seq_len, 2, audio_rep_dim:{72, 1024}] 2 for stereo.
        # y: [batch_size, seq_len]
        # Output
        print("x", x.shape)
        print("y", y.shape)
        print("dims",self.config.audio_rep_dim, self.config.enc_d_model)
        x_emb = self.audio_projection_layer(x) # [batch_size, seq_len, 2, d_model]
        print("x_emb", x_emb.shape)
        x_emb_l = self.positional_encoder(x_emb[:,:,0]) # left : [batch_size, seq_len, d_model] 
        x_emb_r = self.positional_encoder(x_emb[:,:,1]) # right : [batch_size, seq_len, d_model]
        x_emb = torch.cat([x_emb_l, x_emb_r], dim=1) # [batch_size, 2*seq_len, d_model], Concatenate stereo channels.
        enc_output = self.encoder(x_emb) # [batch_size, seq_len, d_model]
        if y is None:
            print("y_target이 없는디요")
            return self.generate(enc_output, strategy=strategy, sample_arg=sample_arg)
        else:
            y_emb = self.midi_embedding_layer(y)
            y_emb = self.positional_encoder(y_emb)
            y_out = self.decoder(y_emb, enc_output)
            midi_logits = self.midi_projection_layer(y_out)
            return midi_logits
    
    def generate(self, enc_output, seq_len=150, strategy="top-p", sample_arg=None):
        # Input
        # enc_output: [batch_size, seq_len, d_model]
        # Output
        # midi_logits: [batch_size, seq_len, midi_vocab_size]
        batch_size = enc_output.shape[0]
        # Start token.
        y = torch.zeros(batch_size, 1).long().cuda()
        y[:,0] = 0
        for i in tqdm(range(seq_len)):
            y_emb = self.midi_embedding_layer(y)
            y_emb = self.positional_encoder(y_emb)
            y_out = self.decoder(y_emb, enc_output)
            midi_logits = self.midi_projection_layer(y_out)
            sampled_token = self.sample(midi_logits[:,i,:], strategy=strategy, sample_arg=sample_arg)
            if torch.all(sampled_token == 2):
                return y
            else:
                y = torch.cat([y, sampled_token], dim=-1)
        return y
    
    def sample(self, logits, strategy="greedy", sample_arg=None):
        # Input
        # logits: [batch_size, midi_vocab_size]
        # Output
        # sampled_token: [batch_size, 1]
        if strategy == "greedy":
            return torch.argmax(logits, dim=-1).unsqueeze(-1)
        elif strategy == "top-k": # This is generated by copilot. Need to check.
            k = sample_arg
            top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
            top_k_probs = self.softmax(top_k_logits)
            sampled_token = torch.multinomial(top_k_probs, 1)
            return top_k_indices.gather(-1, sampled_token)
        elif strategy == "top-p": # This is generated by copilot. Need to check.
            p = sample_arg
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(self.softmax(sorted_logits), dim=-1)
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, -float('inf'))
            sampled_token = torch.multinomial(self.softmax(logits), 1)
            return sampled_token

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
