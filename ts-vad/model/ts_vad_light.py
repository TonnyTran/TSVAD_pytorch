import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules import *
from model.ecapa_tdnn import *

class TS_VAD(nn.Module):
    def __init__(self, args):
        super(TS_VAD, self).__init__()
        # Speaker Encoder
        self.speech_encoder = ECAPA_TDNN(C = 1024)
        loadedState = torch.load('pretrain/ecapa-tdnn.model', map_location="cuda")
        selfState = self.state_dict()
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)

        self.speech_down = nn.Sequential(
                nn.Conv1d(1536, 192, 5, stride=4, padding=2),
                nn.BatchNorm1d(192),
                nn.ReLU(),
            )
        self.backend_down = nn.Sequential(
            nn.Conv1d(384 * 4, 384, 5, stride=1, padding=2),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            )
        self.pos_encoder = PositionalEncoding(384, dropout=0.05)    
        self.pos_encoder_m = PositionalEncoding(384, dropout=0.05)
        self.single_backend = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=384, dim_feedforward = 384 * 4, nhead=4), num_layers=3)
        self.multi_backend = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=384, dim_feedforward = 384 * 4, nhead=4), num_layers=3)

    # B: batchsize, T: number of frames (1 frame = 0.04s)
    # Obtain the reference speech represnetation
    def rs_forward(self, x): # B, 25 * T
        B, T = x.shape
        x = self.speech_encoder.forward(x)
        x = self.speech_down(x)
        x = x[:,:,:(T // 16000) * 25]
        x = x.transpose(1, 2)
        return x

    # Obtain the target speaker represnetation
    def ts_forward(self, x): # B, 4, 16000 * T
        return x

    # Combine for ts-vad results
    def cat_forward(self, rs_embeds, ts_embeds):
        # Extend ts_embeds for time alignemnt
        ts_embeds = ts_embeds.unsqueeze(2) # B, 4, 1, 192
        ts_embeds = ts_embeds.repeat(1, 1, rs_embeds.shape[1], 1) # B, 4, T, 192
        B, _, T, _ = ts_embeds.shape
        # Transformer for single speaker
        cat_embeds = []
        for i in range(4):
            ts_embed = ts_embeds[:, i, :, :] # B, T, 192    
            cat_embed = torch.cat((ts_embed,rs_embeds), 2) # B, T, 192 + B, T, 192 -> B, T, 384   
            cat_embed = cat_embed.transpose(0,1) # B, 384, T    
            cat_embed = self.pos_encoder(cat_embed)
            cat_embed = self.single_backend(cat_embed) # B, 384, T
            cat_embed = cat_embed.transpose(0,1) # B, T, 384
            cat_embeds.append(cat_embed)
        cat_embeds = torch.stack(cat_embeds) # 4, B, T, 384
        cat_embeds = torch.permute(cat_embeds, (1, 0, 3, 2)) # B, 4, 384, T
        # Combine the outputs
        cat_embeds = cat_embeds.reshape((B, -1, T))  # B, 4 * 384, T
        # Downsampling
        cat_embeds = self.backend_down(cat_embeds)  # B, 384, T
        # Transformer for multiple speakers
        cat_embeds = torch.permute(cat_embeds, (2, 0, 1)) # T, B, 384
        cat_embeds = self.pos_encoder_m(cat_embeds)
        cat_embeds = self.multi_backend(cat_embeds) # T, B, 384
        cat_embeds = torch.permute(cat_embeds, (1, 0, 2)) # B, T, 384
        # Results for each speaker
        cat_embeds = cat_embeds.reshape((B, 4, T, -1))  # B, 4, T, 96
        return cat_embeds