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
        self.train_type = args.train_type
        
        # Encoder hparams.
        self.enc_num_layers = 10//args.layer_cut
        self.enc_num_heads = 16//args.dim_cut
        self.enc_d_model = 1024//args.dim_cut
        self.enc_d_ff = 3072
        # Decoder hparams.
        self.dec_num_layers = 10//args.layer_cut
        self.dec_num_heads = 16//args.dim_cut
        self.dec_d_model = 1024//args.dim_cut
        self.dec_d_ff = 3072
        # Dropout hparams.
        self.embed_dropout = 0.1
        self.attn_dropout = 0.1
        self.ff_dropout = 0.1
        # Other hparams.
        self.midi_vocab_size = 1000+128+4+3 ## <start>,<end>,<pad>,<sep> 1000 time bins, 128 velocity bins, 3 drum types. 1134
        self.audio_vocab_size = 1024+1  # 1024 dac tokens + <start>,<end>,<pad>
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
        self.max_target_len = 345+8+345+8+2
        self.padding_loss_lambda = 0.1
        
        if self.train_type not in ['ksh', 'kick', 'snare', 'hihat', 'kshm'] :
            raise ValueError("Invalid train_type")
        

class EncoderDecoderModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_decoder = EncoderDecoder(config)
        # self.loss_function = self.loss_functions = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim+1)] 
        if self.config.train_type == 'kshm':
            self.loss_functions = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim+1)] 
        else: 
            self.total_loss = nn.CrossEntropyLoss()
            self.audio_loss = nn.CrossEntropyLoss()
            self.padding_loss = nn.CrossEntropyLoss()

            self.start_loss = nn.CrossEntropyLoss()
            # self.loss_functions = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim)] 
            # self.padding_loss_functions = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim)] 
            # self.padding_in_loss_functions_l = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim)]
            # self.padding_in_loss_functions_r = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim)]
    def training_step(self, batch, batch_idx):
        padding_loss, audio_losses,start_loss, mean_acc = self.step(batch)

        # self.log("train_total_loss", total_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        # if self.config.train_type == 'kshm':
        self.log("train_padding_loss", padding_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_audio_loss", audio_losses.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)

        self.log("train_start_loss", start_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_mean_acc", mean_acc.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return (start_loss + audio_losses + padding_loss)
    
    def validation_step(self, batch, batch_idx):
        padding_loss, audio_losses,start_loss, mean_acc = self.step(batch)
        # self.log("valid_total_loss", total_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        # if self.config.train_type == 'kshm':
        self.log("valid_padding_loss", padding_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("valid_audio_loss", audio_losses.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)

        self.log("valid_start_loss", start_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("valid_mean_acc", mean_acc.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)

        return (start_loss + audio_losses + padding_loss)
           
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return self.optimizer
      
    def step(self, batch):
        x_l, y, dac_length = batch # x: audio_rep, y: midi_audio_tokens  # x.shape : (4, TODO(431) , 2, 9), y.shape : (4, 10, 1472) 
        y = rearrange(y, 'b v t -> b t v') # (4, 1472, 10)

        pred = self.encoder_decoder(x_l, y[:,:-1]) # (batch_size, seq_len, total_vocab_size) torch.Size([4, 1471, 10396])
        

        if self.config.train_type == 'kshm': #TODO : need to fixed (padding 뒤로 뺌) #kick snare hihat dac +  midi token
            pass
        
        elif self.config.train_type == 'ksh':  
            pass
        
        else: #inst # TODO 이거 먼저 

            # y_pred : torch.Size([4, 2737, 2157]) b, t, v
            # y : torch.Size([4, 2738, 19])
            
            audio_losses = []
            padding_losses = []
            padding_in_losses = []
                        
            total_losses = []

            
            # x_l = x[:,:,0,:].long()
            
            # x_l :     b, 345, d
            # y :       b, 347, d 
            # xy :      b, 692, d
            # pred :    b, 691, d*v
            
            # #### All Tokens ####
            # pred_ = pred.view(pred.shape[0],pred.shape[1],9, 1025)
            # pred_ = rearrange(pred_, 'b s d v -> b v s d')

            # total_loss = self.total_loss(pred_, target.long())
            target = y[:,1:,:]
            
            #### Contents Start ####
            start_losses = []
            for i in range(len(dac_length)):

                start_logits = pred[i,:8+2,:] # s, d*v
                start_logits = start_logits.view(start_logits.shape[0],9, 1025) # s, d, v 
                start_logits = rearrange(start_logits, 's d v -> s v d') # s, v, d
                start_target = target[i,:8+2,:].clone() # s d
                # start_target[1:,][start_target[1:,] == 0] = -100
                start_losses.append(self.start_loss(start_logits, start_target.long()))

            #### Contents #### 
            for i in range(len(dac_length)):
                audio_logits = pred[i,:dac_length[i]+8+1,:]
                audio_logits = audio_logits.view(audio_logits.shape[0],9, 1025)
                audio_logits = rearrange(audio_logits, 's d v -> s v d')
                audio_target = target[i,:dac_length[i]+8+1,:] # s d
                audio_losses.append(self.audio_loss(audio_logits, audio_target.long()))

            #### Only Padding ####
            for i in range(len(dac_length)): #for i range(batch)
                padding_logits = pred[i,dac_length[i]+8:,:]
                padding_logits = padding_logits.view(padding_logits.shape[0],9, 1025)
                padding_logits = rearrange(padding_logits, 's d v -> s v d')
                padding_target = target[i,dac_length[i]+8:,:] # s d 
                if padding_logits.shape[0] != 0: 
                    padding_losses.append(self.padding_loss(padding_logits, padding_target.long()))

            
        padding_loss = sum(padding_losses)/len(padding_losses)
        audio_loss = sum(audio_losses)/len(audio_losses)
        start_loss = sum(start_losses)/len(start_losses)
        
        # Calculate Accuracy.

        audio_accs = []
        for i in range(len(dac_length)):
            acc_logits = pred[i,:dac_length[i]+8+1,:]
            acc_logits = acc_logits.view(acc_logits.shape[0],9, 1025)
            acc_logits = rearrange(acc_logits, 's d v -> s v d')
            acc_target = target[i,:dac_length[i]+8+1,:] # s d
            acc_pred = torch.argmax(acc_logits, dim=1)
            audio_acc = torch.sum(acc_pred == acc_target).float() / torch.numel(acc_target)
            audio_accs.append(audio_acc)

        mean_acc = (sum(audio_accs)) /len(dac_length)



        return padding_loss, audio_loss, start_loss, mean_acc


    def forward(self, batch): # evaluation step
        
        x, y, dac_length= batch
        # x : torch.Size([1, 345, 2, 9])
        # y : torch.Size([1, 9, 355])
        # y : torch.Size([1, 355, 9])
         
        # ## Training Check ##
        # y = rearrange(y, 'b v t -> b t v') # (4, 1472, 10)
        # y_pred = self.encoder_decoder(x, y[:,:-1]) # (batch_size, seq_len, total_vocab_size) torch.Size([4, 1471, 10396])
        
        
        ###### For Inference ######
        y_pred = self.encoder_decoder(x)
        # y_pred : torch.Size([1, 355, 9])    
        

        return y_pred 

class EncoderDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        
        self.positional_encoder = PositionalEncoder(self.config.dec_d_model, self.config.embed_dropout) # 768, 0.1
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.softmax = nn.Softmax(dim=-1)
        # self.embedding = nn.Embedding(1025, 1)
        if self.config.train_type == 'kshm':
            self.dac_projection_layer = nn.Linear(config.dec_d_model, config.midi_vocab_size + config.audio_vocab_size * 9) ##number
        else: 
            self.dac_projection_layer = nn.Linear(config.dec_d_model,  config.audio_vocab_size * 9)
        self.midi_embedding_layer = nn.Embedding(config.midi_vocab_size, config.dec_d_model)
        self.audio_embedding_layer = nn.ModuleList([nn.Embedding(1025, config.dec_d_model) for _ in range(9)]) ##number

    
    def forward(self, x_l, y=None, strategy="greedy", sample_arg=None):
        if self.config.train_type == 'kshm':
            pass
        elif self.config.train_type == 'ksh':
        # x: torch.Size([4, 345, 2, 9])
        # y: torch.Size([4, 1062, 9])
            pass
        else: 
        # Input
        # x: [batch_size, seq_len, 2, audio_rep_dim:9] 2 for stereo.        torch.Size([4, TODO(431), 2, 9])  audio_rep
        # -> x: [batch_size, seq_len, audio_rep_dim] 
        # y: [batch_size, seq_len, 10]                                      torch.Size([4, 1471, 10])   target_midi_audio_token

            #### x Embedding ####
            # x_l = x[:,:,0,:].long() # x: torch.Size([1, 345, 2, 9]) -> x_l : torch.Size([4, 345, 9]) :  stereo to mono

            for i in range(self.config.audio_rep_dim): 
                x_l_tok_embedding = self.audio_embedding_layer[i](x_l[:,:,i])
                if i == 0:
                    x_tok_embedding = x_l_tok_embedding
                x_tok_embedding = x_tok_embedding+ x_l_tok_embedding
            x_emb_l = self.positional_encoder(x_tok_embedding) #[batch_size, seq_len, dec_d_model]
            
            #### Encoder ####
            enc_output = self.encoder(x_emb_l) # [batch_size, seq_len, d_model]:  [4, TODO(2*431), 768] #TODO : concat 이 맘에 안들면 Parallel로 바꿔보기
            
            #### y Generation ####
            if y is None:
                if self.config.train_type == 'ksh':
                    seq_len =  1062
                elif self.config.train_type == 'kick' or self.config.train_type == 'snare' or self.config.train_type == 'hihat':
                    seq_len =  345+8+1
                return self.generate(enc_output, seq_len =  seq_len, strategy=strategy, sample_arg=sample_arg)

            #### y Embedding ####
            if self.config.train_type == 'kshm':
                y_tok_embedding = self.midi_embedding_layer(y[:,:,0]) # torch.Size([4, 1471]) -> torch.Size([4, 1471, 768])

            for i in range(9): 
                if self.config.train_type == 'kshm': #[batch_size, seq_len, 10]        
                    audio_tok_embedding = self.audio_embedding_layer[i](y[:,:,i+1])

                else: # [batch_size, seq_len, 9]        
                    audio_tok_embedding = self.audio_embedding_layer[i](y[:,:,i])  # y[:,:,i] = [batch_size, seq_len, 1] -> torch.Size([batch_size, seq_len, dec_d_model])
                    if i == 0:
                        y_tok_embedding = audio_tok_embedding
                    y_tok_embedding = y_tok_embedding + audio_tok_embedding #TODO: 덧셈이 맞는지 확인 -> 맞다고 하네요. but 다른 방법론도 적용해보자
            y_emb_l = self.positional_encoder(y_tok_embedding) # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model] :[4, 1471, 768]
            
            #### Decoding ####
            y_out = self.decoder(y_emb_l, enc_output) # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]

            audio_logits = self.dac_projection_layer(y_out) # [batch_size, seq_len, (config.midi_vocab_size + )config.audio_vocab_size * 9]


            return audio_logits # y_pred : torch.Size([4, 1471, 10394])


    def generate(self, enc_output, seq_len, strategy='greedy', sample_arg=None):
        # Input
        # enc_output: [batch_size, seq_len, d_model]
        # Output
        # midi_logits: [batch_size, seq_len, midi_vocab_size]
        batch_size = enc_output.shape[0]
        # Start token.
        y = torch.zeros(batch_size, 1, 9).long().cuda()
        end = False
        for i in tqdm(range(seq_len)):
            for j in range(9):
                audio_tok_embedding = self.audio_embedding_layer[j](y[:,:,j]) # torch.Size([1, 1, 384])
                if j == 0:
                    tok_embedding = audio_tok_embedding
                tok_embedding = tok_embedding+ audio_tok_embedding
            tok_embedding = self.positional_encoder(tok_embedding)
            y_out = self.decoder(tok_embedding, enc_output)
            audio_logits = self.dac_projection_layer(y_out) # [batch_size, seq_len(1), VOCAB_SIZE]
            
            # top-p sampling
            sampled_token = self.sample(audio_logits[:,i,:], strategy=strategy, sample_arg=sample_arg) #[batch_size, 1, 9 or 10]
            if torch.all(sampled_token == 0):
                y = torch.cat([y, sampled_token], dim=1)
                end = True
                return y, end
            else:
                y = torch.cat([y, sampled_token], dim=1)

        return y, end

    
    def sample(self, logits, strategy, sample_arg=None):
        # Input
        # logits: [batch_size, midi_vocab_size]
        # Output
        # sampled_token: [batch_size, 1]
        if strategy == "top-p":
            p = 0.9
            sampled_indices =[]
            if self.config.train_type == 'kshm':
                midi_logits = logits[:, :self.config.midi_vocab_size]
                midi_sorted_logits, midi_indices = torch.sort(midi_logits, descending=True)
                midi_prob = torch.softmax(midi_sorted_logits, dim=-1)
                midi_cum_prob = torch.cumsum(midi_prob, dim=-1)
                midi_index = torch.where(midi_cum_prob >= p, torch.ones_like(midi_cum_prob), torch.zeros_like(midi_cum_prob)).argmax(dim=-1)
                midi_sorted_logits[0,midi_index+1:] = -float("inf")
                midi_top_p_prob = torch.softmax(midi_sorted_logits, dim=-1)
                midi_sampled = torch.multinomial(midi_top_p_prob, 1)
                midi_reindex = torch.gather(midi_indices, dim=-1, index=midi_sampled)
                sampled_indices.append(midi_reindex)

            for i in range(9):
                if self.config.train_type == 'kshm':
                    audio_logits = logits[:, self.config.midi_vocab_size + i*self.config.audio_vocab_size : self.config.midi_vocab_size + (i+1)*self.config.audio_vocab_size]
                else:
                    audio_logits = logits[:, i*self.config.audio_vocab_size : (i+1)*self.config.audio_vocab_size]
                audio_sorted_logits, audio_indices = torch.sort(audio_logits, descending=True)
                audio_prob = torch.softmax(audio_sorted_logits, dim=-1)
                audio_cum_prob = torch.cumsum(audio_prob, dim=-1)
                audio_index = torch.where(audio_cum_prob > p, torch.ones_like(audio_cum_prob), torch.zeros_like(audio_cum_prob)).argmax(dim=-1)
                audio_sorted_logits[0,audio_index+1:] = -float("inf")
                audio_top_p_prob = torch.softmax(audio_sorted_logits, dim=-1)
                audio_sampled = torch.multinomial(audio_top_p_prob, 1)
                audio_reindex = torch.gather(audio_indices, dim=-1, index=audio_sampled)
                sampled_indices.append(audio_reindex)

            sampled_indices = torch.cat(sampled_indices, dim=-1).unsqueeze(0)

            return sampled_indices #torch.Size([1, 1, 9])
        if strategy == "greedy":
            # logits : torch.Size([1, 10396])
            sampled_indices =[]
     
            for i in range(9):
                audio_logits = logits[:, i*self.config.audio_vocab_size : (i+1)*self.config.audio_vocab_size] #torch.Size([1, 1029])
                audio_reindex = torch.argmax(audio_logits, dim=-1).unsqueeze(-1)
                sampled_indices.append(audio_reindex)

            sampled_indices = torch.cat(sampled_indices, dim=-1).unsqueeze(0)
            return sampled_indices #torch.Size([1, 1, 9])
        
        # elif strategy == "top-k": # This is generated by copilot. Need to check.
        #     k = sample_arg
        #     top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        #     top_k_probs = self.softmax(top_k_logits)
        #     sampled_token = torch.multinomial(top_k_probs, 1)
        #     return top_k_indices.gather(-1, sampled_token)
        # elif strategy == "top-p": # This is generated by copilot. Need to check.
        #     p = sample_arg
            
        #     sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        #     cumulative_probs = torch.cumsum(self.softmax(sorted_logits), dim=-1)
        #     sorted_indices_to_remove = cumulative_probs > p
        #     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        #     sorted_indices_to_remove[..., 0] = 0
        #     indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        #     logits = logits.masked_fill(indices_to_remove, -float('inf'))
        #     sampled_token = torch.multinomial(self.softmax(logits), 1)
        #     return sampled_token


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
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=config.enc_num_heads, dropout=config.attn_dropout,batch_first=True, norm_first=True)
        # ff_dim = config.enc_d_ff
        # self.ln1 = nn.LayerNorm(embed_dim)
        # self.ln2 = nn.LayerNorm(embed_dim)
        # self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=config.enc_num_heads, dropout=config.attn_dropout, batch_first=True)
        # self.ff = nn.Sequential(
        #     nn.Linear(embed_dim, ff_dim),
        #     nn.GELU(),
        #     nn.Linear(ff_dim, embed_dim),
        #     nn.Dropout(config.ff_dropout),
        # )
    
    def forward(self, x):
        # # training2 # 원본
        # x = self.ln1(x)
        # x = x + self.mha(x,x,x)[0]
        # x = x + self.ff(self.ln2(x))
    
        # # training
        # residual = x
        # x = residual + self.mha(x,x,x)[0]
        # x = self.ln1(x)
        
        # residual = x
        # x = residual + self.ff(x)
        # x= self.ln2(x)
        
        # # 정석으로 바꿔봤다
        x = self.encoder_layer(x)
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
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=config.dec_num_heads, dropout=config.attn_dropout, batch_first=True, norm_first=True)
        # ff_dim = config.dec_d_ff
        
        # self.ln1 = nn.LayerNorm(embed_dim)
        # self.ln2 = nn.LayerNorm(embed_dim)
        # self.ln3 = nn.LayerNorm(embed_dim)
        # self.mha1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=config.dec_num_heads, dropout=config.attn_dropout, batch_first=True)
        # self.mha2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=config.dec_num_heads, dropout=config.attn_dropout, batch_first=True)
        self.register_buffer("causal_mask", torch.triu(torch.ones(config.max_target_len, config.max_target_len).bool(), diagonal=1))
        # self.ff = nn.Sequential(
        #     nn.Linear(embed_dim, ff_dim),
        #     nn.GELU(),
        #     nn.Linear(ff_dim, embed_dim),
        #     nn.Dropout(config.ff_dropout),
        # )
    
    def forward(self, x, enc_output):
        x = self.decoder_layer(x,enc_output,tgt_mask=self.causal_mask[:x.size(1),:x.size(1)] ,tgt_is_causal=True)
        # x = self.ln1(x)
        # x = x + self.mha1(x, x, x, is_causal=True, attn_mask=self.causal_mask[:x.size(1),:x.size(1)])[0]
        # x = x + self.mha2(self.ln2(x), enc_output, enc_output, is_causal=False)[0]
        # x = x + self.ff(self.ln3(x))
        return x    


if __name__ == "__main__":
    print('good')
    # from encoder_decoder_m import EncoderDecoderModule, EncoderDecoderConfig    
    # from dataset import DrumSlayerDataset
    # from torch.utils.data import DataLoader
    # import wandb
    # import lightning.pytorch as pl
    # import os
    # import datetime
    # from lightning.pytorch.loggers import WandbLogger
    # from lightning.pytorch.loggers import TensorBoardLogger
    # import torch
    # from torch.utils.data import Dataset
    # import numpy as np
    # import pretty_midi
    # import glob

    
    # BATCH_SIZE = 2
    # NUM_WORKERS = 0
    # NUM_DEVICES = 0,1,2,3,4
    # # # Set the desired CUDA device number
    # # torch.cuda.set_device(7)
    # # device_number = torch.cuda.current_device()
    # # print(f"CUDA device number: {device_number}")

    # EXP_NAME = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}-lucky-seven"

    # RESUME = False
    # WANDB = False
    # audio_encoding_type = "latents" # "latents", "codes", "z" (dim: 72, 9, 1024)
    # trained_dir = '/workspace/DrumTranscriber/ckpts'
    # # data_dir = '/workspace/DrumSlayer/generated_data/'
    # # data_dir = '/disk2/st_drums/generated_data/'
    # # trained_dir = '/workspace/DrumTranscriber/ckpts'
    # class TestDataset(Dataset):
    #     def __init__(self, max_len=1472, audio_encoding_type = "codes"):
    #         assert audio_encoding_type in ["latents", "codes", "z"] # dim: 72, 9, 1024
    #         self.file_path =  '/disk2/st_drums/generated_data'

    #         self.max_len = max_len
    #         self.encoding_type = audio_encoding_type
    #         if self.encoding_type == 'codes':
    #             self.size = [10, max_len] ##number
    #     def __getitem__(self, idx):
    #         # audio_rep = np.load(self.file_path + f"drum_data_{self.split}/mixed_loops/{idx}_{self.encoding_type}.npy") ## npy 생성 -> preprocess_dac.py
    #         # audio_rep = rearrange(audio_rep, 'c d t -> t c d') # c: channel, d: dim, t: time
    #         audio_rep = np.random.randint(0, 1024, (2, 9, 431))
    #         audio_rep = rearrange(audio_rep, 'c d t -> t c d')
            
    #         # kick_midi =  pretty_midi.PrettyMIDI(self.file_path + f"drum_data_{self.split}/generated_midi/kick_midi/kick_midi_{idx}.midi")
    #         # snare_midi =  pretty_midi.PrettyMIDI(self.file_path + f"drum_data_{self.split}/generated_midi/snare_midi/snare_midi_{idx}.midi")
    #         # hihat_midi = pretty_midi.PrettyMIDI(self.file_path + f"drum_data_{self.split}/generated_midi/hihat_midi/hihat_midi_{idx}.midi")
    #         # midi_tokens = self.tokenize_midi(kick_midi, snare_midi, hihat_midi) # (,152)
            
    #         audio_tokens = np.random.randint(0,1024, (self.size)) # (d:10, t:2737) # TODO : stereo d => 19
    #         audio_tokens = rearrange(audio_tokens, 'd t -> t d') # (2737, 10)
    #         return audio_rep, audio_tokens # midi_tokens

    #     def __len__(self):
    #         return 1000

    # def main():
    #     os.makedirs(f"{trained_dir}/{EXP_NAME}/", exist_ok=True)    
    #     # wandb.init(project="DrumTranscriber", name=EXP_NAME)
    #     max_len = 1472 
    #     audio_encoding_type = 'codes'
    #     train_path = '/disk2/st_drums/generated_data'
    #     train_dataset = TestDataset(max_len, audio_encoding_type)
    #     train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        
    #     valid_dataset = TestDataset(max_len,  audio_encoding_type)
    #     valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    #     config = EncoderDecoderConfig(audio_rep = audio_encoding_type)

    #     model = EncoderDecoderModule(config)
        
    #     # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     #     dirpath=f"{trained_dir}/{EXP_NAME}/",
    #     #     # dirpath=f"/data5/kyungsu/ckpts/DrumSlayer/{EXP_NAME}/",
    #     #     monitor="train_content_loss",
    #     #     mode = "min",
    #     #     every_n_steps=100,
    #     #     filename = "{epoch}-{train_content_loss:.2f}",
    #     #     verbose=True,
    #     #     save_top_k=3,
    #     #     save_last=False,
    #     # )
    #     if WANDB:
    #         logger = WandbLogger(name=EXP_NAME, project="DrumSlayer")
    #         trainer = pl.Trainer(accelerator="gpu", logger=logger, devices=NUM_DEVICES, max_epochs=1, precision='16-mixed', )
    #     else:
    #         logger = TensorBoardLogger(save_dir=f"{trained_dir}/{EXP_NAME}/logs", name=EXP_NAME)
    #         trainer = pl.Trainer(accelerator="gpu", logger=logger, devices=NUM_DEVICES, max_epochs=1, precision='16-mixed',)
        
    #     trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    # # import os
    # # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # main()