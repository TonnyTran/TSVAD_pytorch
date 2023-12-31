import torch
from torch import nn, Tensor
import torch.nn.functional as F
from model.modules import *
from model.WavLM import WavLM, WavLMConfig

class TS_VAD(nn.Module):
    def __init__(self, args):
        super(TS_VAD, self).__init__()
        # Speech Encoder
        checkpoint = torch.load(args.speech_encoder_pretrain, map_location="cuda")
        cfg  = WavLMConfig(checkpoint['cfg'])
        cfg.encoder_layers = 6
        self.speech_encoder = WavLM(cfg)
        self.speech_encoder.train()
        self.speech_encoder.load_state_dict(checkpoint['model'], strict = False)
        self.speech_down = nn.Sequential(
            nn.Conv1d(768, 192, 5, stride=2, padding=2),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            )
        
        # TS-VAD Backend
        self.backend_down = nn.Sequential(
            nn.Conv1d(384 * 8, 384, 5, stride=1, padding=2),  # Change here
            nn.BatchNorm1d(384),
            nn.ReLU(),
            )
        self.pos_encoder = PositionalEncoding(384, dropout=0.05)
        self.pos_encoder_m = PositionalEncoding(384, dropout=0.05)
        self.single_backend = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=384, dim_feedforward = 384 * 4, nhead=8), num_layers=3)
        self.multi_backend = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=384, dim_feedforward = 384 * 4, nhead=8), num_layers=3)

    def rs_forward(self, x): 
        B, _ = x.shape 
        x = self.speech_encoder.extract_features(x)[0]        
        x = x.view(B, -1, 768)  
        x = x.transpose(1,2)
        x = self.speech_down(x)
        x = x.transpose(1,2) 
        return x

    def ts_forward(self, x):  # B, 8, 192  # Change here
        return x

    def cat_forward(self, rs_embeds, ts_embeds):
        ts_embeds = ts_embeds.unsqueeze(2) 
        ts_embeds = ts_embeds.repeat(1, 1, rs_embeds.shape[1], 1) 
        B, _, T, _ = ts_embeds.shape
        cat_embeds = []
        for i in range(8):  # Change here
            ts_embed = ts_embeds[:, i, :, :] 
            cat_embed = torch.cat((ts_embed,rs_embeds), 2) 
            cat_embed = cat_embed.transpose(0,1) 
            cat_embed = self.pos_encoder(cat_embed)
            cat_embed = self.single_backend(cat_embed) 
            cat_embed = cat_embed.transpose(0,1) 
            cat_embeds.append(cat_embed)
        cat_embeds = torch.stack(cat_embeds) 
        cat_embeds = torch.permute(cat_embeds, (1, 0, 3, 2)) 
        cat_embeds = cat_embeds.reshape((B, -1, T))  
        cat_embeds = self.backend_down(cat_embeds)  
        cat_embeds = torch.permute(cat_embeds, (2, 0, 1)) 
        cat_embeds = self.pos_encoder_m(cat_embeds)
        cat_embeds = self.multi_backend(cat_embeds) 
        cat_embeds = torch.permute(cat_embeds, (1, 0, 2)) 
        cat_embeds = cat_embeds.reshape((B, 8, T, -1))  # Change here
        return cat_embeds