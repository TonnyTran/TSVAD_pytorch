import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules import *
from model.WavLM import WavLM, WavLMConfig


class TS_VAD(nn.Module):
    def __init__(self, args):
        super(TS_VAD, self).__init__()

        # Speech Encoder
        checkpoint = torch.load(args.ref_speech_encoder, map_location="cuda")
        cfg  = WavLMConfig(checkpoint['cfg'])
        cfg.encoder_layers = 12
        self.speech_encoder = WavLM(cfg)
        self.speech_encoder.train()
        self.speech_encoder.load_state_dict(checkpoint['model'], strict = False)
        self.speech_down = nn.Sequential(
            nn.Conv1d(768, 256, 5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            )
        
        # TS-VAD Backend
        self.backend_down = nn.Sequential(
            nn.Conv1d(512 * 4, 512, 5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            )
        self.pos_encoder = PositionalEncoding(384, dropout=0.05)
        self.pos_encoder_m = PositionalEncoding(384, dropout=0.05)
        self.single_backend = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, dim_feedforward = 512 * 4, nhead=8), num_layers=3)
        self.multi_backend = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, dim_feedforward = 512 * 4, nhead=8), num_layers=2)

    # B: batchsize, T: number of frames (1 frame = 0.04s)
    # Obtain the reference speech represnetation
    def rs_forward(self, x): # B, 25 * T
        B, _ = x.shape 
        x = self.speech_encoder.extract_features(x)[0]        
        x = x.view(B, -1, 768)  # B, 50 * T, 768
        x = x.transpose(1,2)
        x = self.speech_down(x)
        x = x.transpose(1,2) # B, 25 * T, 256
        return x

    # Obtain the target speaker represnetation
    def ts_forward(self, x): # B, 4, 256
        x = F.normalize(x, p=2, dim=1)
        return x

    # Combine for ts-vad results
    def cat_forward(self, rs_embeds, ts_embeds):
        # Extend ts_embeds for time alignemnt
        ts_embeds = ts_embeds.unsqueeze(2) # B, 4, 1, 256
        ts_embeds = ts_embeds.repeat(1, 1, rs_embeds.shape[1], 1) # B, 4, T, 256
        B, _, T, _ = ts_embeds.shape
        # Transformer for single speaker
        cat_embeds = []
        for i in range(4):
            ts_embed = ts_embeds[:, i, :, :] # B, T, 256
            cat_embed = torch.cat((ts_embed, rs_embeds), 2) # B, T, 256 + B, T, 256 -> B, T, 512
            cat_embed = cat_embed.transpose(0,1) # B, 512, T
            cat_embed = self.pos_encoder(cat_embed)
            cat_embed = self.single_backend(cat_embed) # B, 512, T
            cat_embed = cat_embed.transpose(0,1) # B, T, 512
            cat_embeds.append(cat_embed)
        cat_embeds = torch.stack(cat_embeds) # 4, B, T, 512
        cat_embeds = torch.permute(cat_embeds, (1, 0, 3, 2)) # B, 4, 512, T
        # Combine the outputs
        cat_embeds = cat_embeds.reshape((B, -1, T))  # B, 4 * 512, T
        # Downsampling
        cat_embeds = self.backend_down(cat_embeds)  # B, 512, T
        # Transformer for multiple speakers
        cat_embeds = torch.permute(cat_embeds, (2, 0, 1)) # T, B, 512
        cat_embeds = self.pos_encoder_m(cat_embeds)
        cat_embeds = self.multi_backend(cat_embeds) # T, B, 512
        cat_embeds = torch.permute(cat_embeds, (1, 0, 2)) # B, T, 512
        # Results for each speaker
        cat_embeds = cat_embeds.reshape((B, 4, T, -1))  # B, 4, T, 128
        return cat_embeds
