import torch, math
import torch.nn as nn
import torch.nn.functional as F
from tools import *

class Loss(nn.Module):
    def __init__(self):   
        super(Loss, self).__init__()
        self.fc = nn.Linear(96, 1)
        self.loss = nn.BCEWithLogitsLoss(reduction = 'none')
        self.m = nn.Sigmoid()

    def forward(self, x, labels=None): # x : B, 4, T, 96; labels: B, 4, T
        x = self.fc(x).squeeze(-1)        
        total_loss = 0

        # Remove the loss for the all slience part, slience part: predict 0 directly
        slience_labels = torch.sum(labels, dim = 1)
        slience_labels = slience_labels * torch.ones_like(slience_labels)
        slience_labels = torch.where(slience_labels >= 1, 1, 0)

        num_no_slience = sum(sum(slience_labels))
        for i in range(4):
            output = x[:,i,:]
            label = labels[:,i,:]
            loss = self.loss(output, label)
            wo_slience_loss = slience_labels * loss
            total_loss += sum(sum(wo_slience_loss)) / num_no_slience

        slience_labels = slience_labels.unsqueeze(1)
        slience_labels = slience_labels.repeat(1, 4, 1)

        x = self.m(x)
        x = x * slience_labels
        x = x.data.cpu().numpy()

        return total_loss / 4, x