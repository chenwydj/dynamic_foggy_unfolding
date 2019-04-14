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
import torch.nn.functional as F

from .base import BaseNet
from .fcn import FCNHead
from .danet import DANetHead

__all__ = ['MSN', 'get_msn']
up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class MSN(nn.Module):
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
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(MSN, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        self._up_kwargs = up_kwargs
        self.mse = nn.MSELoss()

        # total inplanes: 64
        # 32 -- 512
        self.backbone0 = BaseNet(nclass, backbone, aux, se_loss, norm_layer=norm_layer, inplanes=64, input_ch=3, **kwargs) # image
        self.backbone1 = BaseNet(nclass, backbone, aux, se_loss, norm_layer=norm_layer, inplanes=64, input_ch=3, **kwargs) # Lab
        self.backbone2 = BaseNet(nclass, backbone, aux, se_loss, norm_layer=norm_layer, inplanes=64, input_ch=19, **kwargs) # seg
        # self.head0 = FCNHead(2048, nclass, norm_layer)
        # self.head1 = FCNHead(2048, nclass, norm_layer)
        # self.head2 = FCNHead(2048, nclass, norm_layer)
        # self.head = DANetHead(2048*3, nclass, norm_layer)

        self.head = FCNHead(2048*3, nclass, norm_layer)
        # self.avgpool = nn.AvgPool2d(7)
        # self.fc = nn.Linear(1000, 1)

    def forward(self, image, Lab, seg):
        _, _, _, c4_image = self.backbone0.base_forward(image)
        _, _, _, c4_Lab = self.backbone1.base_forward(Lab)
        _, _, _, c4_seg = self.backbone2.base_forward(seg)

        output = self.head(torch.cat([c4_image, c4_Lab, c4_seg], dim=1))

        # output = self.fc(self.avgpool(output)) # regress mIoU

        if self.aux and self.training:
            # auxout_image = self.head0(c4_image)
            # auxout_Lab = self.head1(c4_Lab)
            # auxout_seg = self.head2(c4_seg)
            # return output#, auxout_image, auxout_Lab, auxout_seg
            return output
        else:
            return output

    def _crop(self, features, top_lefts, ratios, output_size):
        """
        features: global feature maps, b, c, h, w
        output_size: (h, w)
        """
        b = features.size(0)
        if b != len(top_lefts) and b == 1:
            features = torch.cat([features,]*len(top_lefts))
        b = features.size(0)
        grid = np.zeros((b,) + output_size + (2,), dtype=np.float32)
        gridMap = np.array([[(cnt_w/(output_size[1]-1), cnt_h/(output_size[0]-1)) for cnt_w in range(output_size[1])] for cnt_h in range(output_size[0])])
        for i in range(b):
            x, y = top_lefts[i][1], top_lefts[i][0]
            h, w = ratios[i][0], ratios[i][1]
            bbox = np.array([x, y, x + w, y + h]) * 2 - 1
            grid[i, :, :, 0] = bbox[0] + (bbox[2] - bbox[0])*gridMap[:, :, 0]
            grid[i, :, :, 1] = bbox[1] + (bbox[3] - bbox[1])*gridMap[:, :, 1]
        
        grid = torch.from_numpy(grid).cuda()
        sample = F.grid_sample(features.cuda(), grid)#.cpu().numpy().astype(np.uint8)
        return sample
        
def get_msn(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='~/.encoding/models', **kwargs):
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
    from ..datasets import datasets#, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = MSN(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict= False)
    return model