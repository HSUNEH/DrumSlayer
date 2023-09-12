
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from midi_data import DrumTorchDataset
import lightning as L
import librosa
import numpy as np
import os

# import wandb


# class T5(L.LightningModule):
#     def __init__(self):

#     def forward(self, x):

#     def training_step(self, batch, batch_idx):
    
#     def configure_optimizers(self):

class Autoencoder(L.LightningModule):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.compress_ratio = 2 ** 10 

        self.maxpool_5 = nn.MaxPool1d(5)
        self.maxpool_3 = nn.MaxPool1d(3)
        self.maxpool_2 = nn.MaxPool1d(2)
        self.act = nn.ReLU() 

        self.d_conv0 = nn.Conv1d(1, 50, 3, padding = 1, stride = 1,bias=False)

        self.d_convs = nn.ModuleList(
            [nn.Conv1d(50, 50, 3, padding = 1,stride=1, bias = False) for _ in range(10)]
        )

        self.encode_conv = nn.Conv1d(50,50,3,padding = 1, stride= 1,bias=False)

        self.u_convs = nn.ModuleList([
            nn.Conv1d(50, 50, 3, padding = 1, stride=1, bias = False),
        ]+
        [nn.Conv1d(100, 50, 3, padding = 1, stride=1, bias = False) for _ in range(5)]
        )
        self.last_conv = nn.Conv1d(100, 100, 3, padding = 1, stride=1)
        self.last_conv = nn.Conv1d(100, 100, 3, padding = 1, stride=1)

        self.hidden_size = 3 # 악기 개수
        # self.num_layers = 3 
        self.channel_size = 100 # input / 16
        self.rec_1 = nn.GRU(self.channel_size, self.hidden_size, batch_first=True, bidirectional=True,bias=True)
        self.rec_2 = nn.GRU(self.hidden_size*2, self.hidden_size, batch_first=True, bidirectional=False,bias=True)
        self.rec_3 = nn.GRU(1, 1, batch_first=True, bidirectional=False,bias=False)
        
    def forward(self, x):
        # print("input : ", x.shape)
        # # Padding
        # div = self.compress_ratio
        # nsp_src = x.shape[1]
        # nsp_pad = (div - (nsp_src % div)) % div
        # if nsp_pad != 0:
        #     x = F.pad(x, (0, nsp_pad))
        # print("input padding: ", x.shape)
        x = torch.unsqueeze(x,1)
        x= self.act(self.d_conv0(x)) # ( batch , n_ch , N) 
        # print("d_conv0 : ", x.shape)

        xs = []
        i = 0
        for module in self.d_convs:
            x = module(x)
            x = self.act(x)
            if i < 2:
                x = self.maxpool_5(x)
            else:
                x = self.maxpool_2(x)
            # print("d_conv : ", x.shape)
            xs.append(x)
            i += 1 
        y = self.act(self.encode_conv(xs.pop()))
        
        ys = []
        ys.append(y)
        for module in self.u_convs:
            y = module(y)
            y = self.act(y)
            # print('before interpolate : ',y.shape)
            y = F.interpolate(y, scale_factor=2,
                    mode=int(y.dim() == 4) * 'bi' + 'linear', align_corners=False)
            x = xs.pop()

            y = torch.cat((y, x), dim=1)
            # print("u_conv : ", y.shape)
            ys.append(y)

        r = self.act(self.last_conv(y)) # (batch, 100, rate/16)
        # print("r : ", r.shape)
        r = r.transpose(1,2)
        # print("r (tr): ", r.shape)

        after = self.act(self.rec_1(r)[0]) 

        # print("rec_1 : ", after.shape)

        after = self.act(self.rec_2(after)[0]) 

        # print("rec_2 : ", after.shape)

        b, time, n_insts = after.shape
        after = after.reshape((b * time), 1, n_insts)  # (b * t, 1, n_insts)
        after = after.transpose(1, 2)  # (b * t, n_insts, 1)
        # print("shape changed : ", after.shape)

        after = self.rec_3(after)[0]
        # print("rec_3 : ", after.shape)
        
        after = after.transpose(1, 2).reshape(b, time, n_insts)
        represent = self.act(after.transpose(1, 2))
        # print("shape changed : ", represent.shape)
        

        result_np = torch.zeros([represent.size(0),3,1920], dtype=torch.float32) # TODO : batch, end값 삽입

        upsample_factor = result_np.size(2) // represent.size(2)  # 확장 비율 계산
        for i in range(upsample_factor):
            result_np[:, :, i::upsample_factor] = represent
        # print("result : " , result_np.shape)
        
        return result_np

    def training_step(self, batch, batch_idx):
        x, y_target = batch
        # batch -> torch.Size([batch_size, 96768]) torch.Size([batch_size, 6, 96000])
        y_target = y_target.float()
        y_pred = self(x).to('cuda') #torch.Size([batch_size, 6, 96000])
        criterion  = nn.MSELoss()

        # Onset 개수를 MSE loss 
        onset_loss = (y_target.nonzero().size(0)-y_pred.nonzero().size(0))

        train_loss = onset_loss
        # 
        
        self.log('train_loss', train_loss, prog_bar=True, on_step=False, on_epoch=True)
        # # wandb.log({"train_loss": train_loss})  # Log train_loss to `wandb`
        return train_loss

    def validation_step(self, batch, batch_idx):

        x, y_target = batch
        # batch -> torch.Size([batch_size, 96768]) torch.Size([batch_size, 6, 96000])
        y_target = y_target.float()
        y_pred = self(x).to('cuda') #torch.Size([batch_size, 6, 96000])
        criterion  = nn.MSELoss()
        val_loss = criterion(y_pred, y_target)
        
        # metric = BinaryF1Score()
        # f1score = metric(y_pred, y_target)

        metrics = {"val_loss": val_loss}#, "val_f1score": f1score}
        # self.log_dict(metrics , prog_bar=True, on_step=True, on_epoch=False)
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        # # wandb.log(metrics)  # Log loss to wandb
        return metrics

    def test_step(self, batch, batch_idx):
        x, y_target = batch
        # batch -> torch.Size([batch_size, 96768]) torch.Size([batch_size, 6, 96000])
        y_target = y_target.float()
        y_pred = self(x).to('cuda') #torch.Size([batch_size, 6, 96000])
    
        criterion  = nn.MSELoss()
        loss = criterion(y_pred, y_target)

        y_pred = y_pred.transpose(0,1).cpu().numpy() #torch.Size([3, batch_size, 1920])
        kick, snare, hihat = y_pred # np size : (batch_size,1920)

        # kick_pr, snare_pr, hihat_pr = np.zeros(kick.shape(0),1920,128),np.zeros(snare.shape(0),1920,128),np.zeros(hihat.shape(0),1920,128)
        
        # TODO : kick to kick_pr

        # np 확인
        test_dir = 'midi_2_wav/drum_data_test/test_outputs'
        os.makedirs(test_dir, exist_ok=True)
        kicknps_dir = test_dir + '/kick'
        os.makedirs(kicknps_dir, exist_ok=True)
        snarenps_dir = test_dir + '/snare'
        os.makedirs(snarenps_dir, exist_ok=True)
        hihatnps_dir = test_dir + '/hihat'
        os.makedirs(hihatnps_dir, exist_ok=True)
        i = 0
        for k_np in kick:
            np.save(kicknps_dir+f'/kicknps_{batch_idx*batch_size+i}',k_np)
            i+=1
        i = 0
        for s_np in snare:
            np.save(snarenps_dir+f'/snarenps_{batch_idx*batch_size+i}',s_np)
            i+=1
        i = 0
        for h_np in hihat:
            np.save(hihatnps_dir+f'/hihatnps_{batch_idx*batch_size+i}',h_np)
            i+=1
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

from lightning.pytorch.callbacks import ModelCheckpoint

from lightning.pytorch.callbacks.early_stopping import EarlyStopping


if __name__ == "__main__":
    # # # # # # Initialize wandb
    # wandb.init(project='Beat2Midi')
    top_models_callback = ModelCheckpoint(
    dirpath="model",
    verbose=True,
    every_n_train_steps=100,
    save_top_k=-1,
    save_last=True,
    filename="model-{epoch:02d}-{val_loss:.2f}",
    )
    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.001, patience=50, mode="min")

    # Parameters
    n_mels=128
    hop_length=512 
    n_fft=2048
    sr = 48000
    
    # 데이터 전처리 
    batch_size = 32
    dataloader = DrumTorchDataset('midi_2_wav', batch_size)
    # val_dataloader = DrumTorchDataset('midi_2_wav/drum_data_val', batch_size)

    # 모델 및 Trainer 생성
    model = Autoencoder().to('cuda')
    trainer = L.Trainer(accelerator="gpu", devices= 1, max_epochs=3, callbacks=[top_models_callback, early_stop_callback]) # strategy="ddp", 

    # 학습 시작
    trainer.fit( model, datamodule = dataloader)#, val_dataloader)


    # 테스트
    predictions = trainer.test(datamodule = dataloader)