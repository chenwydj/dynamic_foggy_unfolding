###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample
import numpy as np

from bdd.encoding.models.base import BaseNet
from EnlightenGAN.lib.nn import SynchronizedBatchNorm2d as SynBN2d
from EnlightenGAN.models.attention import PAM_Module, CAM_Module

__all__ = ['FCN', 'get_fcn', 'get_fcn_resnet50_pcontext', 'get_fcn_resnet50_ade']

class FCN(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    Examples
    --------
    >>> model = FCN(nclass=21, backbone='resnet50')
    >>> print(model)
    """
    # def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=SynBN2d, **kwargs):
        super(FCN, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        # self.dahead = DAHead(2048)
        # self.head = FCNHead(2048//4, nclass, norm_layer)
        # self.head_generator = FCNHead(2048//4, 3, norm_layer)
        self.head = FCNHead(2048, nclass, norm_layer)
        self.head_generator = FCNHead(2048, 3, norm_layer)

    def forward(self, input, gray):
        imsize = input.size()[2:]
        _, _, _, c4 = self.base_forward(input)

        # c4 = self.dahead(c4)

        seg = self.head(c4)
        seg = upsample(seg, imsize, **self._up_kwargs)
        # seg_mask = seg.max(dim=1)[0].unsqueeze(1)
        latent = self.head_generator(c4)
        latent = upsample(latent, imsize, **self._up_kwargs)
        # latent = gray * seg_mask * latent
        latent = gray * latent
        output = latent + input
        return output, latent, seg
        
class DAHead(nn.Module):
    def __init__(self, in_channels, norm_layer=SynBN2d):
        super(DAHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv_in = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False), norm_layer(inter_channels), nn.ReLU())
        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.conv_in(x)
        sa_feat = self.sa(x)
        sc_feat = self.sc(x)
        return self.alpha * sa_feat + self.beta * sc_feat + x

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=SynBN2d):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)


def get_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='~/.encoding/models', **kwargs):
    r"""FCN model from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_
    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    Examples
    --------
    >>> model = get_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = FCN(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict= False)
    return model

def get_fcn_resnet50_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_fcn_resnet50_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_fcn('pcontext', 'resnet50', pretrained, root=root, aux=False, **kwargs)

def get_fcn_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_fcn_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_fcn('ade20k', 'resnet50', pretrained, root=root, **kwargs)
