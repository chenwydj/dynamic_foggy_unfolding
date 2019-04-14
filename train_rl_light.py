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
# opt_gan.dataroot = "/ssd1/chenwy/bdd100k/light_enhance_AB/seg_85/trainA"
# opt_gan.name = "single_unet_conv_add_vary_attention_Tresidual_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1_360px_align"
# opt_gan.name = "bdd.seg.day120.night85_G.latent.gamma_seg.10_vgg0.75_360px_align"
# opt_gan.name = "bdd.day100.105.night0.75_G.latent.gamma_vgg1_180px_align"
opt_gan.name = "bdd.seg.day100.105.night0.75_G.latent.gamma_seg.10_vgg0.5_180px_align"
opt_gan.model = "single"
opt_gan.which_direction = "AtoB"
opt_gan.no_dropout = True
opt_gan.dataset_mode = "unaligned"
opt_gan.which_model_netG = "sid_unet_resize"
# opt_gan.which_model_netG = "sid_unet_res_resize"
opt_gan.fineSize = 180
opt_gan.skip = 1
opt_gan.use_norm = 1
opt_gan.use_wgan = 0
opt_gan.self_attention = True
opt_gan.times_residual = True
opt_gan.instance_norm = 0
opt_gan.resize_or_crop = "no"
opt_gan.which_epoch = "400"
opt_gan.nThreads = 1   # test code only supports nThreads = 1
opt_gan.batchSize = 1  # test code only supports batchSize = 1
opt_gan.serial_batches = True  # no shuffle
opt_gan.no_flip = True  # no flip

data_loader = CreateDataLoader(opt_gan)
dataset = data_loader.load_data()
gan = create_model(opt_gan)

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
# opt_seg.ft_resume = "/home/chenwy/DynamicLightEnlighten/bdd/bdd100k_seg/fcn_model/res50_di_360px_daytime/model_best.pth.tar"
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

##### Policy #################################################
from policy import Policy

evaluation = False

lr = 1e-3
gamma = 1
lambd_entropy = 0.1
# policy = Policy(hidden_dim=2, rnn_type='lstm')
policy = Policy(hidden_dim=11, rnn_type=None)

if evaluation:
    checkpoint = torch.load("/home/chenwy/DynamicLightEnlighten/bdd100k_seg/policy_model/image_lstm.vgg.avgpool.argmax_delta.clip1.2.action-mean0.975_entropy.0_gamma.1_lr1e4_update.5_2019-02-01-13-05/model_best.pth.tar")
    policy.load_state_dict(checkpoint['state_dict'])
    policy.eval()
else:
    # params_list = [{'params': policy.vgg.parameters(), 'lr': lr},]
    # params_list.append({'params': policy.rnn.parameters(), 'lr': lr*10})
    params_list = [{'params': policy.resnet.parameters(), 'lr': lr},]

policy = policy.cuda()
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
    _, _, h, w = image.size()
    if h < 193 or w < 193:
        if h > w:
            image = F.interpolate(image, (int(np.round(h * 193. / w)), 193))
        else:
            image = F.interpolate(image, (193, int(np.round(w * 193. / h))))
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


def evaluate(mode="patch"): # mode="origin"|"gan"|"patch"
    policy.eval()
    tbar = tqdm(dataset)
    total_inter = 0; total_union = 0
    total_inter_origin = 0; total_union_origin = 0
    total_niqe = []
    total_niqe_origin = []
    total_n = []
    with torch.no_grad():
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
                idx = total_union > 0
                tbar.set_description('mIoU: %.3f, NIQE: %.3f' % ((1.0 * total_inter[idx] / (np.spacing(1) + total_union[idx])).mean(), np.mean(total_niqe)))
            elif mode == "gan":
                image = transformer(Image.open(data["A_paths"][0])).unsqueeze(0)
                mask = _mask_transform(Image.open(os.path.join("/ssd1/chenwy/bdd100k/seg/labels/", opt_gan.phase, os.path.splitext(data["A_paths"][0].split("/")[-1])[0] + '_train_id.png')))
                data["A"] = image
                r,g,b = data["A"][0, 0, :, :]+1, data["A"][0, 1, :, :]+1, data["A"][0, 2, :, :]+1
                A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2. # h, w
                A_gray = A_gray.unsqueeze(0).unsqueeze(0)
                data["A_gray"] = A_gray
                gan.set_input(data)
                visuals, fake_B_tensor = gan.predict()
                inter, union = get_mIoU(fake_B_tensor, mask, inter_union=True)
                total_inter += inter; total_union += union
                idx = total_union > 0
                niqe = get_niqe(fake_B_tensor)
                total_niqe.append(niqe)
                tbar.set_description('mIoU: %.3f, NIQE: %.3f' % ((1.0 * total_inter[idx] / (np.spacing(1) + total_union[idx])).mean(), np.mean(total_niqe)))
            elif mode == "patch":
                image = transformer(Image.open(data["A_paths"][0])).unsqueeze(0)
                mask = _mask_transform(Image.open(os.path.join("/ssd1/chenwy/bdd100k/seg/labels/", opt_gan.phase, os.path.splitext(data["A_paths"][0].split("/")[-1])[0] + '_train_id.png')))

                inter, union = get_mIoU(image, mask, inter_union=True)
                total_inter_origin += inter; total_union_origin += union
                niqe = get_niqe(image)
                total_niqe_origin.append(niqe)

                crop_size = opt_gan.fineSize
                _, _, h, w = image.size()
                n_h = int(np.ceil(1. * h / crop_size))
                n_w = int(np.ceil(1. * w / crop_size))
                step_h = 1.0 * (h - crop_size) / (n_h - 1)
                step_w = 1.0 * (w - crop_size) / (n_w - 1)
                image_new = torch.zeros(1, 3, h, w)
                template = torch.zeros(1, 3, h, w)

                for i in range(n_w):
                    for j in range(n_h):
                        # x is w, y is h
                        x = int(np.round(i * step_w)); y = int(np.round(j * step_h))
                        data["A"] = image[0:1, :, y: y + crop_size, x: x + crop_size]
                        r,g,b = data["A"][0, 0, :, :]+1, data["A"][0, 1, :, :]+1, data["A"][0, 2, :, :]+1
                        A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2. # h, w
                        A_gray = A_gray.unsqueeze(0).unsqueeze(0)
                        data["A_gray"] = A_gray

                        # GAN prediction ###########################
                        gan.set_input(data)
                        visuals, fake_B_tensor = gan.predict()
                        data = image_cycle(data, visuals)
                        image_new[0:1, :, y: y + crop_size, x: x + crop_size] += data["A"]
                        template[0:1, :, y: y + crop_size, x: x + crop_size] += 1
                image_new /= template
                # evaluation ############################
                inter, union = get_mIoU(image_new, mask, inter_union=True)
                total_inter += inter; total_union += union
                idx = total_union > 0
                niqe = get_niqe(image_new)
                total_niqe.append(niqe)
                tbar.set_description('mIoU_ori: %.3f, mIoU: %.3f, NIQE_ori: %.3f, NIQE: %.3f' % ((\
                    1.0 * total_inter_origin[idx] / (np.spacing(1) + total_union_origin[idx])).mean(),\
                    (1.0 * total_inter[idx] / (np.spacing(1) + total_union[idx])).mean(),\
                    np.mean(total_niqe_origin), np.mean(total_niqe)))
            elif mode == "policy":
                image = transformer(Image.open(data["A_paths"][0])).unsqueeze(0)
                mask = _mask_transform(Image.open(os.path.join("/ssd1/chenwy/bdd100k/seg/labels/", opt_gan.phase, os.path.splitext(data["A_paths"][0].split("/")[-1])[0] + '_train_id.png')))

                inter, union = get_mIoU(image, mask, inter_union=True)
                total_inter_origin += inter; total_union_origin += union
                niqe = get_niqe(image)
                total_niqe_origin.append(niqe)

                crop_size = opt_gan.fineSize
                _, _, h, w = image.size()
                n_h = int(np.ceil(1. * h / crop_size))
                n_w = int(np.ceil(1. * w / crop_size))
                step_h = 1.0 * (h - crop_size) / (n_h - 1)
                step_w = 1.0 * (w - crop_size) / (n_w - 1)
                image_new = torch.zeros(1, 3, h, w)
                template = torch.zeros(1, 3, h, w)

                for i in range(n_w):
                    for j in range(n_h):
                        # x is w, y is h
                        x = int(np.round(i * step_w)); y = int(np.round(j * step_h))
                        data["A"] = image[0:1, :, y: y + crop_size, x: x + crop_size]
                        r,g,b = data["A"][0, 0, :, :]+1, data["A"][0, 1, :, :]+1, data["A"][0, 2, :, :]+1
                        A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2. # h, w
                        A_gray = A_gray.unsqueeze(0).unsqueeze(0)
                        data["A_gray"] = A_gray

                        n = 0
                        # input = torch.cat([data["A"], data["A_gray"]], dim=1).cuda()
                        input = data["A"].cuda()
                        # input = data["A_gray"].cuda()
                        action, dist = policy(input)
                        while action > 0 and n < N_limit:
                            # GAN prediction ###########################
                            gan.set_input(data)
                            visuals, fake_B_tensor = gan.predict()
                            n += 1
                            # GAN reset image data #########################
                            data = image_cycle(data, visuals)
                            # Policy #######################################
                            # input = torch.cat([data["A"], data["A_gray"]], dim=1).cuda()
                            input = data["A"].cuda()
                            # input = data["A_gray"].cuda()
                            action, dist = policy(input)
                        total_n.append(n)
                        image_new[0:1, :, y: y + crop_size, x: x + crop_size] += data["A"]
                        template[0:1, :, y: y + crop_size, x: x + crop_size] += 1
                image_new /= template
                # evaluation ############################
                inter, union = get_mIoU(image_new, mask, inter_union=True)
                total_inter += inter; total_union += union
                idx = total_union > 0
                niqe = get_niqe(image_new)
                total_niqe.append(niqe)
                tbar.set_description('mIoU_ori: %.3f, mIoU: %.3f, NIQE_ori: %.3f, NIQE: %.3f, n_ave: %.1f' % ((\
                    1.0 * total_inter_origin[idx] / (np.spacing(1) + total_union_origin[idx])).mean(),\
                    (1.0 * total_inter[idx] / (np.spacing(1) + total_union[idx])).mean(),\
                    np.mean(total_niqe_origin), np.mean(total_niqe), np.mean(total_n)))


N_limit = 10
update_interval = 5
epochs = 100
reward_history = [0, 0.975] # mean, smooth_ratio

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if evaluation:
    evaluate(mode=evaluation) # "gan"
else:
    # optimizer = torch.optim.SGD(policy.parameters(), lr, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(params_list, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    opt_seg.dataset = "bdd100k_seg"
    opt_seg.model = "policy"
    opt_seg.checkname = "image_lstm.vgg.avgpool.argmax_delta.clip2.5.6.5.action-mean0.975_entropy.0_gamma.1_lr1e3_update.5_" + time.strftime("%Y-%m-%d-%H-%M",time.localtime())
    print(opt_seg.checkname)
    writer = SummaryWriter(log_dir=os.path.join("/home/chenwy/DynamicLightEnlighten/logs", opt_seg.checkname, time.strftime("%Y-%m-%d-%H-%M",time.localtime())))

    for epoch in range(epochs):
        print(opt_seg.checkname)
        reward_history = train(epoch, reward_history)
        # evaluate()

webpage.save()
#####################################################