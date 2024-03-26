import math
import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl
import pl_bolts
from tqdm import tqdm

from einops import rearrange


class InstDecoderConfig:
    def __init__(self, audio_rep, args, **kwargs):
        self.train_type = args.train_type
        
        # # Encoder hparams.
        # self.enc_num_layers = 10//args.layer_cut
        # self.enc_num_heads = 12//args.dim_cut
        # self.enc_d_model = 768//args.dim_cut
        # self.enc_d_ff = 3072
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
        if self.train_type == 'kshm' or self.train_type == 'ksh':
            self.audio_vocab_size = 1024+1 # 1024 dac tokens + <start>,<end>,<pad>,<sep>
        else:
            self.audio_vocab_size = 1024+1   # 1024 dac tokens + <start>,<end>,<pad>
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
        self.max_target_len = 345+355
        self.max_len = 345+354
        self.padding_loss_lambda = 0.1

        if self.train_type not in ['ksh', 'kick', 'snare', 'hihat', 'kshm'] :
            raise ValueError("Invalid train_type")
        

class InstDecoderModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inst_decoder = InstDecoder(config)
        # self.loss_function = self.loss_functions = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim+1)] 
        if self.config.train_type == 'kshm':
            self.loss_functions = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim+1)] 
        else: 
            self.total_loss = nn.CrossEntropyLoss()
            self.audio_loss = nn.CrossEntropyLoss()
            self.padding_loss = nn.CrossEntropyLoss()
            self.dac_loss= nn.CrossEntropyLoss()
            # self.loss_functions = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim)] 
            # self.padding_loss_functions = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim)] 
            # self.padding_in_loss_functions_l = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim)]
            # self.padding_in_loss_functions_r = [nn.CrossEntropyLoss() for _ in range(config.audio_rep_dim)]
    def training_step(self, batch, batch_idx):
        total_loss, padding_loss, audio_losses, dac_losses = self.step(batch)

        self.log("train_total_loss", total_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        # if self.config.train_type == 'kshm':
        self.log("train_padding_loss", padding_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_audio_loss", audio_losses.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_dac_loss", dac_losses.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return (audio_losses + padding_loss)
    
    def validation_step(self, batch, batch_idx):
        total_loss, padding_loss, audio_losses, dac_losses = self.step(batch)
        self.log("valid_total_loss", total_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        # if self.config.train_type == 'kshm':
        self.log("valid_padding_loss", padding_loss.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("valid_audio_loss", audio_losses.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("valid_dac_loss", dac_losses.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return (audio_losses + padding_loss)
           
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return self.optimizer
    
    def step(self, batch):
        x, y, dac_length = batch # x: audio_rep, y: midi_audio_tokens  # x.shape : (4, TODO(431) , 2, 9), y.shape : (4, 10, 1472) 
        y = rearrange(y, 'b v t -> b t v') # (4, 1472, 10)
        pred = self.inst_decoder(x, y[:,:-1]) # (batch_size, seq_len, total_vocab_size) torch.Size([4, 1471, 10396])
        

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

            
            x_l = x[:,:,0,:].long()
            xy = torch.cat((x_l, y[:,:,:]), dim=1)
            
            #### All Tokens ####
            pred_ = pred.view(pred.shape[0],pred.shape[1],9, 1025)
            pred_ = rearrange(pred_, 'b t d v -> b v t d')
            target = xy[:,1:,:].long() # b t d 
            total_loss = self.total_loss(pred_, target.long())
            
            #### Only Contents #### 
            for i in range(len(dac_length)):
                pred_i = pred[i,344:344+dac_length[i],:]
                pred_i = pred_i.reshape((pred_i.shape[0]* 9), 1025) # batch , len, 1028*9 -> batch * len * 9 , 1028
                y_target = xy[i,345:345+dac_length[i],:]
                y_target = y_target.reshape(-1) # batch, len, 9 -> batch*len *9
                audio_losses.append(self.audio_loss(pred_i, y_target.long()))
            
            #### Only Padding ####
            for i in range(len(dac_length)): #for i range(batch)
                padding_logits = pred[i,dac_length[i]+1+345:,:].reshape((pred.shape[1]-(dac_length[i]+1+345))*9,1025)
                padding_target = xy[i,dac_length[i]+1+345+1:,:].reshape(((xy.shape[1]-(dac_length[i]+1+345+1))* 9)).long()
                padding_losses.append(self.padding_loss(padding_logits, padding_target))
                # if padding_logits.shape[0] != 0: 
                #     padding_losses.append(self.padding_loss(padding_logits, padding_target))
                # len(audio_losses) = 4 / len(padding_losses) = 4
            
            #### Audio Dac ####
            dac_losses = []
            dac_logits = pred[:,:345,:]
            dac_logits = dac_logits.reshape((dac_logits.shape[0]* dac_logits.shape[1]* 9), 1025)
            dac_target = xy[:,1:345+1,:].reshape(-1)
            
            dac_losses.append(self.dac_loss(dac_logits, dac_target.long()))
            
        padding_loss = sum(padding_losses)/len(padding_losses)
        audio_loss = sum(audio_losses)/len(audio_losses)
        dac_loss = sum(dac_losses)/len(dac_losses)
        
        return total_loss, padding_loss, audio_loss, dac_loss    

    def forward(self, batch): # evaluation step
        
        x, y, dac_length= batch
        y = rearrange(y, 'b v t -> b t v') # (4, 1472, 10)
        # x : torch.Size([1, 345, 2, 9])
        # y : torch.Size([1, 9, 355])
        # y : torch.Size([1, 355, 9])
         
        # ## Training Check ##
        # y = rearrange(y, 'b v t -> b t v') # (4, 1472, 10)
        # y_pred = self.inst_decoder(x, y[:,:-1]) # (batch_size, seq_len, total_vocab_size) torch.Size([4, 1471, 10396])
        

        ###### For Inference ######
        y_pred,end = self.inst_decoder(x)
        # y_pred : torch.Size([1, 355, 9])


        return (y_pred,end)

class InstDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.positional_encoder = PositionalEncoder(self.config.dec_d_model, self.config.embed_dropout) # 768, 0.1

        # self.decoder = Decoder(config)
        # self.softmax = nn.Softmax(dim=-1)
        # self.embedding = nn.Embedding(1025, 1)
        if self.config.train_type == 'kshm':
            self.dac_projection_layer = nn.Linear(config.dec_d_model, config.midi_vocab_size + config.audio_vocab_size * 9) ##number
        else: 
            self.dac_projection_layer = nn.Linear(config.dec_d_model,  config.audio_vocab_size * 9)
        self.midi_embedding_layer = nn.Embedding(config.midi_vocab_size, config.dec_d_model)
        self.audio_embedding_layer = nn.ModuleList([nn.Embedding(1025, config.dec_d_model) for _ in range(9)]) 
        self.dropout = nn.Dropout(config.embed_dropout)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.dec_num_layers)]
        )
        self.ln = nn.LayerNorm(config.dec_d_model)
        self.fc = nn.Linear(config.dec_d_model, 9*config.audio_vocab_size)
    def forward(self, x, y=None, strategy="greedy", sample_arg=None):
        if self.config.train_type == 'kshm':
            pass
        elif self.config.train_type == 'ksh':
        # x: torch.Size([4, 345, 2, 9])
        # y: torch.Size([4, 1062, 9])
            pass
        else: 
        # Input
        # x: [batch_size, 345, 2, audio_rep_dim:9] 2 for stereo.        torch.Size([4, TODO(431), 2, 9])  audio_rep
        # -> x: [batch_size, seq_len, audio_rep_dim] 
        # y: [batch_size, 354, 10]                                      torch.Size([4, 1471, 10])   target_midi_audio_token

            x_l = x[:,:,0,:].long() # x: torch.Size([1, 345, 2, 9]) -> x_l : torch.Size([4, 345, 9]) :  stereo to mono
            ### inference ###
            if y is None:
                seq_len =  355
                return self.generate(x_l, seq_len =  seq_len, strategy=strategy, sample_arg=sample_arg)
            
            
            ### train ###
            for i in range(9): 
                x_tok_embedding = self.audio_embedding_layer[i](x_l[:,:,i])  # 
                y_tok_embedding = self.audio_embedding_layer[i](y[:,:,i])  # y[:,:,i] = [batch_size, seq_len, 1] -> torch.Size([batch_size, seq_len, dec_d_model])
                if i == 0:
                    audio_tok_embedding = x_tok_embedding
                    inst_tok_embedding = y_tok_embedding
                audio_tok_embedding = audio_tok_embedding + x_tok_embedding
                inst_tok_embedding = inst_tok_embedding + y_tok_embedding #TODO: 덧셈이 맞는지 확인 -> 맞다고 하네요. but 다른 방법론도 적용해보자

            embedding = torch.cat((audio_tok_embedding, inst_tok_embedding), dim =1) 

            # TODO : positional encoding
            
            # tok_embedding.shape == (batch_size, seq_len, embed_dim)
            output = self.dropout(embedding)
            output = self.blocks(output)
            output = self.ln(output)
            output = self.fc(output) #torch.Size([16, 699, 9252])

            # output.shape == (batch_size, seq_len, vocab_size)
            return output


    def generate(self, x_l, seq_len, strategy='top-p', sample_arg=None):
        # Input
        # enc_output: [batch_size, seq_len, d_model]
        # Output
        # midi_logits: [batch_size, seq_len, midi_vocab_size]
        batch_size = x_l.shape[0]

        # Start token.
        end = False
        for i in tqdm(range(seq_len)):
            for j in range(9): 
                x_tok_embedding = self.audio_embedding_layer[j](x_l[:,:,j])  # 
                if j == 0:
                    audio_tok_embedding = x_tok_embedding
                audio_tok_embedding = audio_tok_embedding + x_tok_embedding
            embedding = audio_tok_embedding #torch.cat((audio_tok_embedding, torch.zeros(batch_size, 1, 9).long()), dim =1) 

            output = self.dropout(embedding)
            output = self.blocks(output)
            output = self.ln(output)
            output = self.fc(output) #torch.Size([16, 699, 9252])
            
            # top-p sampling
            sampled_token = self.sample(output[:,344+i,:], strategy=strategy, sample_arg=sample_arg) #[batch_size, 1, 9 or 10]
            if torch.all(sampled_token == 1):
                x_l = torch.cat([x_l, sampled_token], dim=1)
                end = True
                return x_l, end
            else:
                x_l = torch.cat([x_l, sampled_token], dim=1)

        return x_l, end
    
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

  
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.dec_d_model
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
        embed_dim = config.dec_d_model
        self.num_heads = config.dec_num_heads
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


# class Encoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.enc_num_layers)])
    
#     def forward(self, x):
#         # Input
#         # x: [batch_size, seq_len, d_model]
#         # Output
#         # x: [batch_size, seq_len, d_model]
#         for layer in self.layers:
#             x = layer(x)
   
#         return x

# class EncoderLayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.config = config
#         embed_dim = config.enc_d_model
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=config.enc_num_heads, dropout=config.attn_dropout,batch_first=True, norm_first=True)
#         # ff_dim = config.enc_d_ff
#         # self.ln1 = nn.LayerNorm(embed_dim)
#         # self.ln2 = nn.LayerNorm(embed_dim)
#         # self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=config.enc_num_heads, dropout=config.attn_dropout, batch_first=True)
#         # self.ff = nn.Sequential(
#         #     nn.Linear(embed_dim, ff_dim),
#         #     nn.GELU(),
#         #     nn.Linear(ff_dim, embed_dim),
#         #     nn.Dropout(config.ff_dropout),
#         # )
    
#     def forward(self, x):
#         # # training2 # 원본
#         # x = self.ln1(x)
#         # x = x + self.mha(x,x,x)[0]
#         # x = x + self.ff(self.ln2(x))
    
#         # # training
#         # residual = x
#         # x = residual + self.mha(x,x,x)[0]
#         # x = self.ln1(x)
        
#         # residual = x
#         # x = residual + self.ff(x)
#         # x= self.ln2(x)
        
#         # # 정석으로 바꿔봤다
#         x = self.encoder_layer(x)
#         return x

# class Decoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.dec_num_layers)])
    
#     def forward(self, x, enc_output):
#         # Input
#         # x: [batch_size, seq_len, d_model]
#         # enc_output: [batch_size, seq_len, d_model]
#         # Output
#         # x: [batch_size, seq_len, d_model]
#         for layer in self.layers:
#             x = layer(x, enc_output)
#         return x

# class DecoderLayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         embed_dim = config.dec_d_model
#         self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=config.dec_num_heads, dropout=config.attn_dropout, batch_first=True, norm_first=True)
#         # ff_dim = config.dec_d_ff
        
#         # self.ln1 = nn.LayerNorm(embed_dim)
#         # self.ln2 = nn.LayerNorm(embed_dim)
#         # self.ln3 = nn.LayerNorm(embed_dim)
#         # self.mha1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=config.dec_num_heads, dropout=config.attn_dropout, batch_first=True)
#         # self.mha2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=config.dec_num_heads, dropout=config.attn_dropout, batch_first=True)
#         self.register_buffer("causal_mask", torch.triu(torch.ones(config.max_target_len, config.max_target_len).bool(), diagonal=1))
#         # self.ff = nn.Sequential(
#         #     nn.Linear(embed_dim, ff_dim),
#         #     nn.GELU(),
#         #     nn.Linear(ff_dim, embed_dim),
#         #     nn.Dropout(config.ff_dropout),
#         # )
    
#     def forward(self, x, enc_output):
#         x = self.decoder_layer(x,enc_output,tgt_mask=self.causal_mask[:x.size(1),:x.size(1)] ,tgt_is_causal=True)
#         # x = self.ln1(x)
#         # x = x + self.mha1(x, x, x, is_causal=True, attn_mask=self.causal_mask[:x.size(1),:x.size(1)])[0]
#         # x = x + self.mha2(self.ln2(x), enc_output, enc_output, is_causal=False)[0]
#         # x = x + self.ff(self.ln3(x))
#         return x    


if __name__ == "__main__":
    print('good')
