import torch 
import torch.nn as nn


audio_projection_layer = nn.Sequential(
            nn.Linear(9, 768),
            nn.GELU(),
            nn.Linear(768, 768)
        )

x= torch.rand(4,431,2,9)

x_emb = audio_projection_layer(x)

breakpoint()