import torch, math
import torch.nn as nn
import torch.nn.functional as F
from tools import *

class Loss(nn.Module):
    def __init__(self):   
        super(Loss, self).__init__()
        self.fc = nn.Linear(48, 1)  # Change input size to 48
        self.loss = nn.BCEWithLogitsLoss(reduction = 'mean')
        self.m = nn.Sigmoid()

    def forward(self, x, labels=None):  # x : B, 8, T, 48; labels: B, 8, T
        x = self.fc(x).squeeze(-1)  # Remove last dimension after applying fc
        total_loss = 0

        for i in range(8):
            output = x[:,i,:]
            label = labels[:,i,:]
            loss = self.loss(output, label)
            total_loss += loss
            
        x = self.m(x)
        x = x.data.cpu().numpy()

        return total_loss / 8, x