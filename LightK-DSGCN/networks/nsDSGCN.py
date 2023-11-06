################ Ablation study: without spatial module ################

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from modules import FilterLinear

# choose device
USE_CUDA = torch.cuda.is_available()
if not USE_CUDA:
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE)

class tconv_encode(nn.Module):
    """
    Performed on batch_size*detector*7 data, to merge the time features of previous days.
    Output is batch_size*detector*1
    """
    def __init__(self, feature_size):
        super(tconv_encode, self).__init__()

        self.feature_size = feature_size # number of detectors

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 3), stride=(1, 1), dilation=(1, 1), padding=(0, 1)),
            nn.ReLU(),
            nn.Dropout(p=0.2))
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 1), dilation=(1, 2), padding=(0, 2)),
            nn.ReLU(),
            nn.Dropout(p=0.2))
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(1, 3), stride=(1, 1), dilation=(1, 4), padding=(0, 4)),
            nn.ReLU(),
            nn.Dropout(p=0.2))
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=(1, 3), stride=(1, 1), dilation=(1, 8), padding=(0, 8)),
            nn.ReLU())

    def forward(self, input):
        x = input
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


class linear_encode(nn.Module):
    """
    Performed on batch_size*detector*7 data, to merge the time features of previous days.
    Output is batch_size*detector*1
    """
    def __init__(self, feature_size, pred_size):
        super(linear_encode, self).__init__()

        self.feature_size = feature_size # number of detectors
        self.pred_size = pred_size  # number of time slots to be predicted

        self.block0 = nn.Sequential(
            nn.Linear(1, pred_size),
            nn.Sigmoid())

        self.block1 = nn.Sequential(
            nn.Linear(feature_size, feature_size//8),
            nn.Sigmoid(),
            nn.Dropout(p=0.2))
        self.block2 = nn.Sequential(
            nn.Linear(feature_size//8, feature_size//8),
            nn.Sigmoid(),
            nn.Dropout(p=0.2))
        self.block3 = nn.Sequential(
            nn.Linear(feature_size//8, 2))

    def forward(self, input):
        # x = self.block0(input)
        x = self.block1(input) # batch*node
        x = self.block2(x)
        x = self.block3(x)# batch*time
        return x


class nsDSGCN(nn.Module):

    def __init__(self, K, A, feature_size, pred_size=1, ext_feature_size = 4, Clamp_A=True):
        # GC-LSTM
        '''
        Args:
            K: K-hop graph
            A: adjacency matrix
            feature_size: number of nodes
            pred_size: the length of output
            ext_feature_size: number of extra features. Only include periodical features by default
            Clamp_A: Boolean value, clamping all elements of A between 0. to 1.
        '''
        super(nsDSGCN, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size
        self.ext_feature_size = ext_feature_size
        self.pred_size = pred_size

        hidden_size = self.feature_size

        # RNN
        input_size = self.feature_size

        self.tconv = tconv_encode(feature_size=self.feature_size) ################### timestep convolution

        self.rfl = nn.Linear(input_size + hidden_size*2, hidden_size) # input: tcn output, seasonal, pred and residual
        self.ril = nn.Linear(input_size + hidden_size*2, hidden_size)
        self.rol = nn.Linear(input_size + hidden_size*2, hidden_size)
        self.rCl = nn.Linear(input_size + hidden_size*2, hidden_size)

        # multiple output linear layer
        self.fc_mo = linear_encode(self.feature_size, self.pred_size)

    def forward(self, input_t, rHidden_State, rCell_State, pred):
        # LSTM
        rcombined = torch.cat((input_t, rHidden_State, pred), 1)
        rf = torch.sigmoid(self.rfl(rcombined))
        ri = torch.sigmoid(self.ril(rcombined))
        ro = torch.sigmoid(self.rol(rcombined))
        rC = torch.tanh(self.rCl(rcombined))
        rCell_State = rf * rCell_State + ri * rC
        rHidden_State = ro * torch.tanh(rCell_State)

        pred = rHidden_State[:, :self.hidden_size]

        return rHidden_State, rCell_State, pred

    def Bi_torch(self, a):
        a[a < 0] = 0
        a[a > 0] = 1
        return a

    def loop(self, inputs):
        # input: torch.Size([batch_size, seq_len, n_detector])
        batch_size = inputs.size(0)

        time_step = inputs.size(1)
        rHidden_State, rCell_State = self.initHidden(batch_size)

        inputs_t = self.tconv(inputs.permute(0, 2, 1).unsqueeze(1)) # batch_size*time*detector -> batch_size*detector*time
        inputs_t = inputs_t.squeeze(1).permute(0, 2, 1)

        # print('inputs_t', inputs_t.size())

        pred = rHidden_State.clone() # init Kalman filtering result
        for i in range(time_step):
            rHidden_State, rCell_State, pred = self.forward(
                torch.squeeze(inputs_t[:, i:i + 1, :]), rHidden_State, rCell_State, pred)

        # print('pred in loop', pred.size())
        pred = self.fc_mo(pred)

        return pred

    def initHidden(self, batch_size):
        rHidden_State = Variable(torch.zeros(batch_size, self.hidden_size).to(DEVICE))
        rCell_State = Variable(torch.zeros(batch_size, self.hidden_size).to(DEVICE))
        return rHidden_State, rCell_State

    def reinitHidden(self, batch_size, Hidden_State_data, Cell_State_data):
        rHidden_State = Variable(Hidden_State_data.to(DEVICE), requires_grad=True)
        rCell_State = Variable(Cell_State_data.to(DEVICE), requires_grad=True)
        return rHidden_State, rCell_State
