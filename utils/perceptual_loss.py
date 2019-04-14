'''
https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py
https://github.com/NVlabs/PL4NN/blob/master/src/loss.py
https://github.com/tensorflow/models/blob/master/research/compression/image_encoder/msssim.py

m = pytorch_msssim.MSSSIM()
img1 = Variable(torch.rand(1, 1, 256, 256))
img2 = Variable(torch.rand(1, 1, 256, 256))
print(pytorch_msssim.msssim(img1, img2))
print(m(img1, img2))
'''
    
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

import torch
import torch.nn.functional as F
from math import exp
import numpy as np


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1, sigma=2):
    '''
    channel: #out_channel(s)
    '''
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # use 3 as input_channel for RGB images
    window = _2D_window.expand(channel, 3, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, weight_map=None, sigma=2, size_average=True, full=False, val_range=None):
    '''
    weight_map is pixel-wise weight for calculating ssim loss (1 - ret)
    '''
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=1, sigma=sigma).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd)#, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd)#, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd) - mu1_sq#, groups=channel)
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd) - mu2_sq#, groups=channel)
    sigma12 = F.conv2d(img1 * img2, window, padding=padd) - mu1_mu2#, groups=channel)

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # old definitions #####################################
        # v1 = 2.0 * sigma12 + C2
        # v2 = sigma1_sq + sigma2_sq + C2
        # cs = torch.mean(v1 / v2) # contrast sensitivity
        # ssim_map = 1 - ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2) # "1 - l*cs" as the ssim loss

    l = (2. * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    cs = (2. * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2) # contrast sensitivity
    ssim_map = 1 - l * cs # "1 - l*cs" as the ssim loss

    if weight_map is not None:
        weight_map = F.upsample(weight_map, size=(ssim_map.size(2), ssim_map.size(3)), mode='bilinear')
        ssim_map *= weight_map

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1) # get sample(batch)-wise individual means => (1, b)

    if full:
        return ret, l, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, sigma=2, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.sigma = sigma

        # Assume 1 as output channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2, weight_map=None):
        # (_, channel, _, _) = img1.size()

        # if channel == self.channel and self.window.dtype == img1.dtype:
        #     window = self.window
        # else:
        #     window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
        #     self.window = window
        #     self.channel = channel

        # return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
        return ssim(img1, img2, window_size=self.window_size, window=self.window, weight_map=weight_map, size_average=self.size_average)


def msssim(img1, img2, window_size=11, weight_map=None, size_average=True, val_range=None, normalize=False, sigmas=[0.5, 1., 2., 4., 8.]):
    device = img1.device
    ml = []
    mcs = []
    for sigma in sigmas:
        _, l, cs = ssim(img1, img2, window_size=window_size, weight_map=weight_map, sigma=sigma, size_average=size_average, full=True, val_range=val_range)
        ml.append(l)
        mcs.append(cs)

    ml = torch.stack(ml, dim=0)
    mcs = torch.stack(mcs, dim=0)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        ml = (ml + 1) / 2
        mcs = (mcs + 1) / 2

    Pcs = torch.prod(mcs, 0)
    ssim_map = 1 - l[-1] * Pcs # "1 - l*cs" as the ssim loss

    if weight_map is not None:
        weight_map = F.upsample(weight_map, size=(ssim_map.size(2), ssim_map.size(3)), mode='bilinear')
        ssim_map = ssim_map * weight_map

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1) # get sample(batch)-wise individual means => (1, b)

    return ret


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2, weight_map=None):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, weight_map=weight_map, size_average=self.size_average)


# deprecated origin code
    # def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False, sigmas=[0.5, 1., 2., 4., 8.]):
    #     device = img1.device
    #     weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    #     levels = weights.size()[0]
    #     mssim = []
    #     mcs = []
    #     for _ in range(levels):
    #         sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
    #         mssim.append(sim)
    #         mcs.append(cs)
    # 
    #         img1 = F.avg_pool2d(img1, (2, 2))
    #         img2 = F.avg_pool2d(img2, (2, 2))
    # 
    #     mssim = torch.stack(mssim)
    #     mcs = torch.stack(mcs)
    # 
    #     # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    #     if normalize:
    #         mssim = (mssim + 1) / 2
    #         mcs = (mcs + 1) / 2
    # 
    #     pow1 = mcs ** weights
    #     pow2 = mssim ** weights
    #     # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    #     output = torch.prod(pow1[:-1] * pow2[-1])
    #     return output