import torch, math
import torch.nn as nn
import torch.nn.functional as F
from tools import *

class Loss(nn.Module):
    def __init__(self):   
        super(Loss, self).__init__()
        self.fc = nn.Linear(96, 1)
        self.loss = nn.BCEWithLogitsLoss(reduction = 'mean')
        self.m = nn.Sigmoid()

    def forward(self, x, labels=None): # x : B, 4, T, 96; labels: B, 4, T
        x = self.fc(x).squeeze(-1)        
        total_loss = 0

        for i in range(4):
            output = x[:,i,:]
            label = labels[:,i,:]
            loss = self.loss(output, label)
            total_loss += loss
            
        x = self.m(x)
        x = x.data.cpu().numpy()

        return total_loss / 4, x