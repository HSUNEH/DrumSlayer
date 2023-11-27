import torch
import numpy as np
from encodec import EncodecModel
from encodec.utils import convert_audio

class CFGSchduler():
    def __init__(self, init_proba:int=1.0, init_embedding_scale:int=1.0, 
    start_epoch:int=100000, end_epoch:int=200000, proba_ratio:int=0.999, scale_ratio:int=1.001):
        self.mask_proba = init_proba
        self.embedding_scale = init_embedding_scale
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.proba_ratio = proba_ratio
        self.scale_ratio = scale_ratio
        
    def update(self, epoch):
        if self.start_epoch <= epoch <= self.end_epoch:
                if 0.0 < self.mask_proba*self.proba_ratio:
                    self.mask_proba *= self.proba_ratio
                if self.embedding_scale*self.scale_ratio <= 15.0:
                    self.embedding_scale *= self.scale_ratio


def audio_trim_and_pad(audio, length, mono:bool):
    if mono:
        if len(audio) > length:
            audio = audio[0:length]
        elif len(audio) < length:
            audio = np.pad(audio, (0, length - len(audio)), 'constant', constant_values = 0)
    else:
        if len(audio) != 2:
            audio = [audio, audio]
        for i in range(len(audio[0])):
            if audio[0][i] != 0.0 or audio[1][i] != 0.0:
                if i != 0:
                    audio = [audio[0][i:], audio[1][i:]]
                    break
                else:
                    break
        if len(audio[0]) > length:
            audio = np.array([audio[0][0:length], audio[1][0:length]])
        elif len(audio[0]) < length:
            audio = np.array(
                [np.pad(audio[0], (0, length - len(audio[0])), 'constant', constant_values = 0),
                 np.pad(audio[1], (0, length - len(audio[0])), 'constant', constant_values = 0)]
                 )
    return audio


def RMS_parallel(audios, mono:bool):
    if mono:
        return np.sqrt(np.sum(np.multiply(audios, audios), axis=1)/len(audios[0]))
    else:
        return np.mean(np.sqrt(np.sum(np.multiply(audios, audios), axis=2)/len(audios[0][0])), axis=1)


@torch.no_grad()
def audiotolatent(audio, sample_rate, encodecmodel, device):
    if sample_rate != encodecmodel.sample_rate:
        audio = convert_audio(audio, sample_rate, encodecmodel.sample_rate, encodecmodel.channels)
    if audio.dim() != 3:
        audio = torch.unsqueeze(audio, dim=0)
    with torch.no_grad():
        audio_frames = encodecmodel.encode(audio.to(device))
    audio_embedding = torch.cat([encoded[2] for encoded in audio_frames], dim=-1)
    i=0
    for batch in range(audio_frames[0][0].shape[0]):
        j=0
        for encoded in audio_frames:
            if j==0:
                audio_scale = encoded[1][batch].repeat(128, encoded[2].shape[-1])
            else:
                audio_scale = torch.cat([audio_scale, encoded[1][batch].repeat(128, encoded[2].shape[-1])], dim=-1)
            j+=1
        if i==0:
            audio_scales = torch.unsqueeze(audio_scale, dim=0)
        else:
            audio_scales = torch.cat([audio_scales, torch.unsqueeze(audio_scale, dim=0)], dim=0)
        i+=1
    audio_latent = torch.cat([audio_embedding, audio_scales], dim=1)
    return audio_latent


@torch.no_grad()
def SSlatenttoaudio(audio_latent, encodecmodel):
    audio_frames = [
            (None, torch.mean(audio_latent[:, 128:256, :150]), audio_latent[:, :128, :150]),
            (None, torch.mean(audio_latent[:, 128:256, 150:300]), audio_latent[:, :128, 150:300]),
            (None, torch.mean(audio_latent[:, 128:256, 300:450]), audio_latent[:, :128, 300:450]),
            (None, torch.mean(audio_latent[:, 128:256, 450:512]), audio_latent[:, :128, 450:512])
            ]
    return encodecmodel.decode(audio_frames)


def SSslatenttoaudio(audio_latents, encodecmodel):
    kick_recon = SSlatenttoaudio(torch.squeeze(audio_latents[:, 0, :, :], dim=1), encodecmodel=encodecmodel)
    snare_recon = SSlatenttoaudio(torch.squeeze(audio_latents[:, 1, :, :], dim=1), encodecmodel=encodecmodel)
    hhclosed_recon = SSlatenttoaudio(torch.squeeze(audio_latents[:, 2, :, :], dim=1), encodecmodel=encodecmodel)
    hhopen_recon = SSlatenttoaudio(torch.squeeze(audio_latents[:, 3, :, :], dim=1), encodecmodel=encodecmodel)
    return torch.stack((kick_recon, snare_recon, hhclosed_recon, hhopen_recon), dim=1)

def SSslatenttoaudio_DAE(audio_latents, model):
    kick_recon = model.decode(torch.squeeze(audio_latents[:, 0, :, :], dim=1), num_steps=100)
    snare_recon = model.decode(torch.squeeze(audio_latents[:, 1, :, :], dim=1), num_steps=100)
    hhclosed_recon = model.decode(torch.squeeze(audio_latents[:, 2, :, :], dim=1), num_steps=100)
    hhopen_recon = model.decode(torch.squeeze(audio_latents[:, 3, :, :], dim=1), num_steps=100)
    return torch.stack((kick_recon, snare_recon, hhclosed_recon, hhopen_recon), dim=1)


class audio_trim_and_pad_for_calculate_maximum_sample_len():
    def __init__(self):
        self.maximum_length = 0

    def doit(self, audio, length, mono:bool):
        if mono:
            if len(audio) > length:
                audio = audio[0:length]
            elif len(audio) < length:
                audio = np.pad(audio, (0, length - len(audio)), 'constant', constant_values = 0)
        else:
            if len(audio) != 2:
                audio = [audio, audio]
            
            self.maximum_length = max(self.maximum_length, len(audio[0]))

            if len(audio[0]) > length:
                audio = np.array([audio[0][0:length], audio[1][0:length]])
            elif len(audio[0]) < length:
                audio = np.array(
                    [np.pad(audio[0], (0, length - len(audio[0])), 'constant', constant_values = 0),
                    np.pad(audio[1], (0, length - len(audio[0])), 'constant', constant_values = 0)]
                    )
        return audio