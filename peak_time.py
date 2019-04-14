"""
Training file for HRL stage. Support Pytorch 3.0 and multiple GPUs.
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from PIL import Image
import math
import numpy as np
import os
import shutil
import argparse
import time
import logging
import json
from collections import OrderedDict
# import itertools
import models
import sys
import pdb
from tqdm import tqdm
from skvideo import measure


cudnn.benchmark = True


##### EnlightenGAN ################################################
from EnlightenGAN.options.test_options import TestOptions
from EnlightenGAN.data.data_loader import CreateDataLoader
from EnlightenGAN.models.models import create_model
from EnlightenGAN.psnr import test_psnr
import EnlightenGAN.util.util as utils_gan
from EnlightenGAN.util.util import tensor2im, save_image
from EnlightenGAN.util.visualizer import Visualizer
from EnlightenGAN.util import html

opt_gan = TestOptions().parse()
opt_gan.dataroot = "/ssd1/chenwy/bdd100k/seg/images/train"
# opt_gan.dataroot = "/ssd1/chenwy/bdd100k/seg_daytime/images/val"
# opt_gan.name = "single_unet_conv_add_vary_attention_Tresidual_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1_360px_align"
# opt_gan.name = "bdd.seg.day120.night85_G.latent.gamma_seg.10_vgg0.75_360px_align"
opt_gan.name = "bdd.day100.105.night0.75_G.latent.gamma_vgg1_180px_align"
# opt_gan.name = "bdd.seg.day100.105.night0.75_G.latent.gamma_seg.10_vgg0.5_180px_align"
opt_gan.model = "single"
opt_gan.which_direction = "AtoB"
opt_gan.no_dropout = True
opt_gan.dataset_mode = "unaligned"
opt_gan.which_model_netG = "sid_unet_resize"
opt_gan.fineSize = 360
opt_gan.skip = 1
opt_gan.use_norm = 1
opt_gan.use_wgan = 0
opt_gan.self_attention = True
opt_gan.times_residual = True
opt_gan.instance_norm = 0
opt_gan.resize_or_crop = "no"
opt_gan.which_epoch = "200"
opt_gan.nThreads = 1   # test code only supports nThreads = 1
opt_gan.batchSize = 1  # test code only supports batchSize = 1
opt_gan.serial_batches = True  # no shuffle
opt_gan.no_flip = True  # no flip

data_loader = CreateDataLoader(opt_gan)
dataset = data_loader.load_data()
gan = create_model(opt_gan)
# gan.eval()

visualizer = Visualizer(opt_gan)
# create website
web_dir = os.path.join("./ablation/", opt_gan.name, '%s_%s' % (opt_gan.phase, opt_gan.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt_gan.name, opt_gan.phase, opt_gan.which_epoch))
######################################################

##### BDD Seg. #################################################
import bdd.encoding.utils as utils_seg
from bdd.encoding.parallel import DataParallelModel
from bdd.encoding.models import get_segmentation_model
from bdd.experiments.segmentation.option import Options

opt_seg = Options().parse()
opt_seg.dataset = "bdd100k_seg"
opt_seg.model = "fcn"
opt_seg.backbone = "resnet50"
opt_seg.dilated = True
opt_seg.ft = True
opt_seg.ft_resume = "/home/chenwy/DynamicLightEnlighten/bdd/bdd100k_seg/fcn_model/res50_di_180px_L100.255/model_best.pth.tar"
opt_seg.eval = True

seg = get_segmentation_model(opt_seg.model, dataset=opt_seg.dataset, backbone=opt_seg.backbone, aux=opt_seg.aux, se_loss=opt_seg.se_loss,
                               dilated=opt_seg.dilated,
                               # norm_layer=BatchNorm2d, # for multi-gpu
                               base_size=720, crop_size=180, multi_grid=opt_seg.multi_grid, multi_dilation=opt_seg.multi_dilation)
seg = DataParallelModel(seg).cuda()
seg.eval()
if opt_seg.ft:
    checkpoint = torch.load(opt_seg.ft_resume)
    seg.module.load_state_dict(checkpoint['state_dict'], strict=False)
    # self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.ft_resume, checkpoint['epoch']))
####################################################


def _mask_transform(mask):
    target = np.array(mask).astype('int32')
    target[target == 255] = -1
    return torch.from_numpy(target).long()

def image_cycle(data, visuals):
    '''
    batch is ONE!?
    '''
    # visuals["fake_B"]is h, w, c
    A_img = transformer(visuals["fake_B"]) # c, h, w
    r,g,b = A_img[0]+1, A_img[1]+1, A_img[2]+1
    A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2. # h, w
    A_img = A_img.unsqueeze(0)
    A_gray = A_gray.unsqueeze(0).unsqueeze(0)
    data["A"] = A_img
    data["A_gray"] = A_gray
    return data

def get_reward(scores, action, threshold=0.5, reward=1):
    # return 1 if want policy return high prob (enter enlightening).
    # return -1 if want policy return low prob (stop enlightening).
    if action > 0:
        delta = float(scores[-1] - scores[-2])
        delta = float(np.clip(delta, -2.5, 6.5))
    else:
        delta = float(scores[-2] - scores[-1])
        delta = float(np.clip(delta, -6.5, 2.5))
    # if delta > threshold:
    #     # this round of enlighting is good
    #     return reward
    # elif delta < -threshold:
    #     # this round of enlighting is bad
    #     return -reward
    # else: return 0
    return delta

def get_niqe(image):
    minimum_size = 200
    _, _, h, w = image.size()
    if h < minimum_size or w < minimum_size:
        if h > w:
            image = F.interpolate(image, (int(np.round(h * minimum_size / w)), minimum_size))
        else:
            image = F.interpolate(image, (minimum_size, int(np.round(w * minimum_size / h))))
    image = image.detach().cpu().numpy()[0]
    image = (image + 1) / 2 * 255
    r,g,b = image[0], image[1], image[2]
    i = (0.299*r+0.587*g+0.114*b)
    return float(measure.niqe(i))

def get_mIoU(image, target, inter_union=False):
    '''
    image: already transfered by 0.5/0.5
    '''
    outputs = seg(image)
    pred = outputs[0]
    inter, union = utils_seg.batch_intersection_union(pred.data, target, 19)
    # total_inter += inter
    # total_union += union
    if inter_union:
        return inter, union
    else:
        idx = union > 0
        IoU = 1.0 * inter[idx] / (np.spacing(1) + union[idx])
        mIoU = IoU.mean()
        return float(np.nan_to_num(mIoU))

def get_score(image, target):
    # NIQE + Seg. mIoU
    mIoU = get_mIoU(image, target)
    niqe = get_niqe(image)
    # print(mIoU, niqe_value)
    return mIoU * 100 - niqe

def get_luminance(image):
    image = np.array(image)
    r,g,b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    value = 0.299*r+0.587*g+0.114*b
    return float(value.mean())

def train(epoch, reward_history):
    optimizer.zero_grad()
    policy.train()
    scheduler.step()
    # dataset = data_loader._reinit_dataset()
    tbar = tqdm(dataset)
    total_loss = 0; P = TP = N = TN = 0.001
    for i, data in enumerate(tbar):
        policy.init_hidden(batch_size=1)
        # policy.repackage_hidden()
        '''
        data["A"] is (b, 3, h, w)
        data["A_gray"] is (b, 1, h, w)
        '''
        # init score
        Rewards = []; Dists = []; Actions = []
        scores = [get_score(data["A"], data["mask"])]

        # action is a list of prob.
        # input = torch.cat([data["A"], data["A_gray"]], dim=1).cuda()
        input = data["A"].cuda()
        # input = data["A_gray"].cuda()
        action, dist = policy(input)
        Dists.append(dist); Actions.append(action)

        gan.set_input(data)
        # visuals = OrderedDict([('real_A', utils_gan.tensor2im(data["A"])), ('fake_B', utils_gan.tensor2im(data["B"]))]) # in case not go into enlighten
        while action > 0 and len(scores) <= N_limit:
            # GAN prediction ###########################
            visuals, fake_B_tensor = gan.predict()
            # Seg & NIQE reward ######################################
            scores.append(get_score(fake_B_tensor, data["mask"]))
            # Reward ####################################
            reward = get_reward(scores, action)
            reward_history[0] = reward_history[0] * reward_history[1] + reward * (1 - reward_history[1])
            Rewards.append(gamma * (reward - reward_history[0]))
            tbar.set_description('policy: %s, action: %s, reward: %.1f, reward_history: %.1f, #Enlighten: %d' % (str(dist.probs.data), str(action.data), reward, reward_history[0], len(scores) - 1))
            # GAN reset image data #########################
            data = image_cycle(data, visuals)
            gan.set_input(data)
            # Policy ########################################
            # input = torch.cat([data["A"], data["A_gray"]], dim=1).cuda()
            input = data["A"].cuda()
            # input = data["A_gray"].cuda()
            action, dist = policy(input)
            Dists.append(dist); Actions.append(action)
        # GAN prediction ###########################
        visuals, fake_B_tensor = gan.predict()
        # Seg & NIQE reward ######################################
        scores.append(get_score(fake_B_tensor, data["mask"]))
        # Reward ####################################
        reward = get_reward(scores, action)
        reward_history[0] = reward_history[0] * reward_history[1] + reward * (1 - reward_history[1])
        Rewards.append(gamma * (reward - reward_history[0]))
        tbar.set_description('policy: %s, action: %s, reward: %.1f, reward_history: %.1f, #Enlighten: %d' % (str(dist.probs.data), str(action.data), reward, reward_history[0], len(scores) - 1))

        # back-propagate the hybrid loss
        loss = 0
        for idx in range(len(Rewards)):
            # loss += (lambd_entropy * Dists[idx].entropy())
            loss += (- Dists[idx].log_prob(Actions[idx]) * Rewards[idx])
            if Rewards[idx] > 0:
                P += 1
                if Actions[idx] > 0: TP += 1
            elif Rewards[idx] < 0:
                N += 1
                if Actions[idx] < 1: TN += 1
            else: pass
            # print(Dists[idx].entropy(), - Dists[idx].log_prob(Actions[idx]) * Rewards[idx])
        torch.autograd.backward(loss)

        total_loss += loss.data.detach().cpu().numpy()
        if i > 0 and (i + 1) % update_interval == 0:
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalars('IoU', {'loss': loss.data.detach().cpu().numpy(), 'TP': 1. * TP / P, 'TN': 1. * TN / N}, epoch * len(tbar) + i)
            total_loss = 0; P = TP = N = TN = 0.001

        img_path = gan.get_image_paths()
        # print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)

        # if args.gate_type == 'rnn':
        #     # for memory efficiency
        #     hidden = repackage_hidden(hidden)

    utils_seg.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': policy.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'best_pred': self.best_pred,
    }, opt_seg, True)
    return reward_history


N_limit = 10

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

mode="patch" # "gan"
Lum2mIoUs = {}
Lum2NIQEs = {}

tbar = tqdm(dataset)
total_inter = 0; total_union = 0
total_inter_origin = 0; total_union_origin = 0
total_niqe = []
total_niqe_origin = []
total_n = []
total_prob = []
for i, data in enumerate(tbar):
    '''
    data["A"] is (b, 3, h, w)
    data["A_gray"] is (b, 1, h, w)
    '''
    if mode == "origin":
        image = transformer(Image.open(data["A_paths"][0])).unsqueeze(0)
        mask = _mask_transform(Image.open(os.path.join("/ssd1/chenwy/bdd100k/seg/labels/", opt_gan.phase, os.path.splitext(data["A_paths"][0].split("/")[-1])[0] + '_train_id.png')))
        # inter, union = get_mIoU(data["A"], data["mask"], inter_union=True)
        inter, union = get_mIoU(image, mask, inter_union=True)
        total_inter += inter; total_union += union
        # niqe = get_niqe(data["A"])
        niqe = get_niqe(image)
        total_niqe.append(niqe)
        tbar.set_description('mIoU: %.3f, NIQE: %.3f' % ((1.0 * total_inter / (np.spacing(1) + total_union)).mean(), np.mean(total_niqe)))
    elif mode == "gan":
        image = Image.open(data["A_paths"][0])
        luminance = get_luminance(image)
        while luminance in Lum2mIoUs:
            luminance += 0.0001
        image = transformer(image).unsqueeze(0)
        mask = _mask_transform(Image.open(os.path.join("/ssd1/chenwy/bdd100k/seg/labels/", opt_gan.phase, os.path.splitext(data["A_paths"][0].split("/")[-1])[0] + '_train_id.png')))
        mIoU = get_mIoU(image, mask)
        niqe = get_niqe(image)
        Lum2mIoUs[luminance] = [mIoU]; Lum2NIQEs[luminance] = [niqe]

        data["A"] = image
        r,g,b = data["A"][0, 0, :, :]+1, data["A"][0, 1, :, :]+1, data["A"][0, 2, :, :]+1
        A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2. # h, w
        A_gray = A_gray.unsqueeze(0).unsqueeze(0)
        data["A_gray"] = A_gray

        for n in range(N_limit):
            gan.set_input(data)
            visuals, fake_B_tensor = gan.predict()
            mIoU = get_mIoU(fake_B_tensor, mask)
            niqe = get_niqe(fake_B_tensor)
            Lum2mIoUs[luminance].append(mIoU); Lum2NIQEs[luminance].append(niqe)
            data = image_cycle(data, visuals)
        np.save("Lum2mIoUs_" + str(N_limit), Lum2mIoUs)
        np.save("Lum2NIQEs_" + str(N_limit), Lum2NIQEs)
    elif mode == "patch":
        image_ori = Image.open(data["A_paths"][0])
        mask_ori = Image.open(os.path.join("/ssd1/chenwy/bdd100k/seg/labels/", opt_gan.phase, os.path.splitext(data["A_paths"][0].split("/")[-1])[0] + '_train_id.png'))

        crop_size = opt_gan.fineSize
        w, h = image_ori.size
        n_h = int(np.ceil(1. * h / crop_size))
        n_w = int(np.ceil(1. * w / crop_size))
        step_h = 1.0 * (h - crop_size) / (n_h - 1)
        step_w = 1.0 * (w - crop_size) / (n_w - 1)

        for j in range(n_h):
            for i in range(n_w):
                # x is w, y is h
                x = int(np.round(i * step_w)); y = int(np.round(j * step_h))
                image = image_ori.crop((x, y, x+crop_size, y+crop_size))
                mask = mask_ori.crop((x, y, x+crop_size, y+crop_size))
                luminance = get_luminance(image)
                while luminance in Lum2mIoUs:
                    luminance += 0.0001
                image = transformer(image).unsqueeze(0)
                mask = _mask_transform(mask)
                mIoU = get_mIoU(image, mask)
                niqe = get_niqe(image)
                Lum2mIoUs[luminance] = [mIoU]; Lum2NIQEs[luminance] = [niqe]

                data["A"] = image
                r,g,b = data["A"][0, 0, :, :]+1, data["A"][0, 1, :, :]+1, data["A"][0, 2, :, :]+1
                A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2. # h, w
                A_gray = A_gray.unsqueeze(0).unsqueeze(0)
                data["A_gray"] = A_gray

                for n in range(N_limit):
                    gan.set_input(data)
                    visuals, fake_B_tensor = gan.predict()
                    mIoU = get_mIoU(fake_B_tensor, mask)
                    niqe = get_niqe(fake_B_tensor)
                    Lum2mIoUs[luminance].append(mIoU); Lum2NIQEs[luminance].append(niqe)
                    data = image_cycle(data, visuals)
        np.save("Lum2mIoUs_patch180_" + str(N_limit), Lum2mIoUs)
        np.save("Lum2NIQEs_patch180_" + str(N_limit), Lum2NIQEs)