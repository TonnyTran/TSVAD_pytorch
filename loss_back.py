import torch, math
import torch.nn as nn
import torch.nn.functional as F
from tools import *
from sklearn.metrics import average_precision_score

class Loss(nn.Module):
    def __init__(self):   
        super(Loss, self).__init__()
        self.fc = nn.Linear(96, 1)
        self.loss = nn.BCEWithLogitsLoss()
        self.m = nn.Sigmoid()

    def forward(self, x, labels=None): # x : B, 4, T, 96; labels: B, 4, T
        x = self.fc(x).squeeze(-1)        
        total_loss, total_ap = 0, 0
        B, _, T = labels.shape

        MS = 0
        FA = 0
        SC = 0
        correct = 0
        for i in range(4):
            output = x[:,i,:]
            label = labels[:,i,:]
            loss = self.loss(output, label)
            # ap = average_precision_score(label.detach().cpu().numpy().reshape((-1)), self.m(output.detach()).cpu().numpy().reshape((-1)))
            
            total_loss += loss

        for i in range(B):
            for j in range(T):
                loss = self.loss(x[i,:,j], labels[i,:,j])
                label_this = labels[i,:,j].detach().cpu().numpy()
                predict_this = x[i,:,j].detach().cpu().numpy()
                # print(predict_this, x[i,:,j], label_this)
                predict_this = numpy.where(predict_this > 0.0, 1, 0)
                if label_this[0] == predict_this[0] and label_this[1] == predict_this[1] and label_this[2] == predict_this[2] and label_this[3] == predict_this[3]:
                    correct += 1
                else:
                    if  label_this[0] == 0 and label_this[1] == 0 and label_this[2] == 0 and label_this[3] == 0: # Incorrect, this frame is no-speech, False Alarm
                        FA += 1
                    else:
                        if predict_this[0] == 0 and predict_this[1] == 0 and predict_this[2] == 0 and predict_this[3] == 0: # Incorrect, this frame is predicted as no-speech, Miss Speech
                            MS += 1
                        else: # In correct, both label and prediction contains speech, Speaker Confusion
                            SC += 1
                
                # print(correct, FA, MS, SC)
                total_loss += loss
        DER = (MS+FA+SC) / (MS+FA+SC+correct)
        MS = MS / (MS+FA+SC+correct)
        FA = FA / (MS+FA+SC+correct)
        SC = SC / (MS+FA+SC+correct)
        return total_loss / B / T, DER, MS, FA, SC