#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.autograd as autograd
from torch.autograd.variable import Variable
from threading import Lock
from torch.distributions import Categorical
from .convlstm import ConvLSTM
from .vgg import vgg16_bn
from .resnet import resnet50
from .fcn import FCN
from .msn import MSN
from EnlightenGAN.lib.nn import SynchronizedBatchNorm2d as SynBN2d


global_lock = Lock()

# ======================
# Recurrent Gate  Design
# ======================

def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


# =============================
# Recurrent Gate Model with RL
# =============================

class Policy(nn.Module):
    def __init__(self, hidden_dim=4, input_dim=3, rnn_type='lstm'):
        super(Policy, self).__init__()

        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # self.vgg = vgg16_bn(pretrained=True)
        # self.resnet = resnet50(pretrained=True, num_classes=hidden_dim, dilated=True, norm_layer=SynBN2d, root="./", multi_grid=False, multi_dilation=None)
        # self.fcn = FCN(hidden_dim, backbone='resnet50', root="./", dilated=True, norm_layer=SynBN2d, multi_grid=False, multi_dilation=None, input_ch=input_dim)
        self.msn = MSN(hidden_dim, backbone='resnet50', root="./", dilated=True, norm_layer=SynBN2d, multi_grid=False, multi_dilation=None)

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(512, hidden_dim)
        elif self.rnn_type == 'convlstm':
            self.rnn = ConvLSTM(
                input_size=(360, 360),
                input_dim=input_dim,
                hidden_dim=[hidden_dim, hidden_dim, hidden_dim],
                kernel_size=(3, 3),
                num_layers=3,
                batch_first=True,
                bias=True,
                return_all_layers=False)
        else:
            self.rnn = None

        self.hidden = None

    def init_hidden(self, batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.hidden = (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda()),
                       autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda()))

    def repackage_hidden(self):
        self.hidden = repackage_hidden(self.hidden)

    # def forward(self, x):
    def forward(self, image, Lab, seg):
        if self.rnn_type == 'lstm': self.rnn.flatten_parameters()
        
        # x = self.resnet(x)
        # probs = F.softmax(x, dim=1)

        # x = self.fcn(x)[0]
        x = self.msn(image, Lab, seg)
        x = x.view(x.size(0), self.hidden_dim, -1).mean(2)
        probs = F.softmax(x, dim=1)

        # x = self.vgg(x)
        # b, _, h, w = x.size()
        # x = F.avg_pool2d(x, (h, w))
        # out, self.hidden = self.rnn(x.view(1, b, -1), self.hidden) # out: (1, b, 2)
        # # out = out.squeeze() # remove 1-size dimensions: (seq_len, batch, num_directions * hidden_size) => (b, hidden_dim)
        # out = out[0] # remove the seq-len dim
        # probs = F.softmax(out, dim=1)

        # do action selection in the forward pass
        if self.training:
            dist = Categorical(probs)
            action = dist.sample()
            action_infer = probs.argmax()
            return action, dist, action_infer
        else:
            dist = Categorical(probs) # None
            action = probs.argmax()
            return action, dist
        # action_reshape = action.view(action.size(0), 1, 1, 1).float()