import torch
import torch.nn as nn
from model.speakerEncoder import *
from model.resnet import *
from model.WavLM import WavLM, WavLMConfig

class TS_VAD(nn.Module):
    def __init__(self, args):
        super(TS_VAD, self).__init__()
        # Speaker Encoder
        self.speaker_encoder = ResNet()
        self.speaker_encoder.train()
        loadedState = torch.load(args.speaker_encoder, map_location="cuda")
        selfState = self.state_dict()
        for name, param in loadedState.items():
            origname = name
            name = 'speaker_encoder.' + name
            if name not in selfState:
                print("%s is not in the model."%origname)
                continue
            if selfState[name].size() != loadedState[origname].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, selfState[name].size(), loadedState[origname].size()))
                continue
            selfState[name].copy_(param)
        for param in self.speaker_encoder.parameters():
            param.requires_grad = False

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
        x = x.transpose(1,2) # B, 25 * T, 192
        return x

    # Obtain the target speaker represnetation
    def ts_forward(self, x): # B, 4, 80, T * 100 
        B, _, D, T = x.shape
        x = x.view(B*4, D, T)
        x = self.speaker_encoder.forward(x)
        x = x.view(B, 4, -1) # B, 4, 192
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
        cat_embeds = self.multi_backend(cat_embeds) # T, B, 384
        cat_embeds = torch.permute(cat_embeds, (1, 0, 2)) # B, T, 384
        # Results for each speaker
        cat_embeds = cat_embeds.reshape((B, 4, T, -1))  # B, 4, T, 96
        return cat_embeds
