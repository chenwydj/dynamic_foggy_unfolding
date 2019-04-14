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
from skimage import color, feature

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
evaluation = 'origin'


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
opt_gan.dataroot = "/ssd1/chenwy/cityscapes/"
opt_gan.name = "cityscapes.seg.3.focal3_G.multitask.seg_D.layer4.3_vgg0_300px_align"
opt_gan.model = "single"
opt_gan.which_direction = "AtoB"
opt_gan.no_dropout = True
opt_gan.dataset_mode = "unaligned"
opt_gan.which_model_netG = "sid_unet_res_resize"
opt_gan.fineSize = 500
opt_gan.skip = 1
opt_gan.use_norm = 1
opt_gan.use_wgan = 0
opt_gan.self_attention = True
opt_gan.times_residual = True
opt_gan.instance_norm = 0
opt_gan.resize_or_crop = "no"
opt_gan.which_epoch = "latest"
opt_gan.nThreads = 1 # test code only supports nThreads = 1
opt_gan.serial_batches = True  # no shuffle
opt_gan.no_flip = True  # no flip


if evaluation:
    opt_gan.batchSize = 1 # test code only supports batchSize = 1
else:
    opt_gan.batchSize = 5

data_loader = CreateDataLoader(opt_gan)
dataset = data_loader.load_data()
# gan = create_model(opt_gan)

# visualizer = Visualizer(opt_gan)
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
opt_seg.dataset = "cityscapes"
opt_seg.model = "fcn"
opt_seg.backbone = "resnet50"
opt_seg.dilated = True
opt_seg.ft = True
opt_seg.ft_resume = "/home/chenwy/DynamicDeHaze/cityscapes/fcn_model/res50_di_mg_500px/model_best.pth.tar"
opt_seg.eval = True

seg = get_segmentation_model(opt_seg.model, dataset=opt_seg.dataset, backbone=opt_seg.backbone, aux=opt_seg.aux, se_loss=opt_seg.se_loss,
                               dilated=opt_seg.dilated,
                               # norm_layer=BatchNorm2d, # for multi-gpu
                               base_size=720, crop_size=500, multi_grid=True, multi_dilation=[2,4,8])
# seg = DataParallelModel(seg).cuda()
seg = torch.nn.DataParallel(seg).cuda()
seg.eval()
if opt_seg.ft:
    checkpoint = torch.load(opt_seg.ft_resume)
    seg.module.load_state_dict(checkpoint['state_dict'], strict=False)
    # self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.ft_resume, checkpoint['epoch']))
####################################################

##### Policy #################################################
from models.policy import Policy

lr = 3e-4
gamma = 1
lambd_entropy = 0.3
# hidden_dim: # action choices
# policy = Policy(hidden_dim=2, rnn_type='lstm')
policy = Policy(hidden_dim=4, input_dim=23, rnn_type=None)
r_neg = 5; r_pos = 5

if evaluation == "policy":
    # checkpoint = torch.load("/home/chenwy/DynamicLightEnlighten/bdd100k_seg/policy_model/image.Lab.seg_0.75_msn.act4_adapt20.0.55.clip5.5_entropy0.3_lr3e3.800epoch_2019-04-02-03-48/model_best.pth.tar")
    # policy.load_state_dict(checkpoint['state_dict'])
    policy.eval()
else:
    # params_list = [{'params': policy.vgg.parameters(), 'lr': lr},]
    # params_list.append({'params': policy.rnn.parameters(), 'lr': lr*10})
    # params_list = [{'params': policy.resnet.parameters(), 'lr': lr},]
    ##################################
    # params_list = [{'params': policy.fcn.pretrained.parameters(), 'lr': lr*10},
    #                 {'params': policy.fcn.head.parameters(), 'lr': lr*10}]
    ##################################
    params_list = [{'params': policy.msn.parameters(), 'lr': lr*10}]
    ##################################

policy = policy.cuda()
policy = nn.DataParallel(policy)
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

def get_niqe(image):
    minimum_size = 200
    _, _, h, w = image.size()
    if h < minimum_size or w < minimum_size:
        if h > w:
            image = F.interpolate(image, (int(np.round(h * minimum_size / w)), minimum_size), mode="bilinear")
        else:
            image = F.interpolate(image, (minimum_size, int(np.round(w * minimum_size / h))), mode="bilinear")
    image = image.detach().cpu().numpy()[0]
    image = (image + 1) / 2 * 255
    r,g,b = image[0], image[1], image[2]
    i = (0.299*r+0.587*g+0.114*b)
    return float(measure.niqe(i))

def get_mIoU_from_softmax(softmax, target, inter_union=False):
    inter, union = utils_seg.batch_intersection_union(softmax, target, 19)
    # total_inter += inter
    # total_union += union
    if inter_union:
        return inter, union
    else:
        idx = union > 0
        IoU = 1.0 * inter[idx] / (np.spacing(1) + union[idx])
        mIoU = IoU.mean()
        return float(np.nan_to_num(mIoU))

def get_mIoU(image, target, inter_union=False):
    '''
    image: already transfered by 0.5/0.5
    '''
    outputs = seg(image)
    pred = outputs[0]
    return get_mIoU_from_softmax(pred.data, target, inter_union=inter_union)

def get_score(image, target):
    # NIQE + Seg. mIoU
    mIoU = get_mIoU(image, target)
    niqe = get_niqe(image)
    # return float(mIoU * 100 - niqe)
    return float(mIoU * 100)

def get_reward(scores, threshold=0.5, reward=1):
    delta = float(scores[-1] - scores[-2])
    delta = float(np.clip(delta, -2.5, 6.5))
    return delta

def train(epoch, reward_history, r_neg, r_pos):
    optimizer.zero_grad()
    policy.train()
    scheduler.step()
    # dataset = data_loader._reinit_dataset()
    tbar = tqdm(dataset)
    total_loss = 0; P = N = 0; mIoU_gain = []; niqe_gain = []
    rewards = torch.zeros(opt_gan.batchSize).cuda()
    total_dists = torch.empty(0, 4).cuda()
    for i, data in enumerate(tbar):
        # policy.init_hidden(batch_size=1)
        # policy.repackage_hidden()
        '''
        data["A"] is (b, 3, h, w)
        data["A_gray"] is (b, 1, h, w)
        '''
        # init score
        rewards.resize_(data["A"].size(0)).copy_(torch.zeros(data["A"].size(0)))

        # action is a list of prob. ##########################
        # input = data["A_gray_border"].cuda()
        # input = data["A_border"].cuda()
        # input = torch.cat([data["A_border"], data["A_gray_border"]], dim=1).cuda()
        # input = torch.cat([data["A_border"].cuda(), data["A_gray_border"].cuda(), seg(data["A_border"].cuda())[0]], dim=1)
        # actions, dists, _ = policy(input)
        with torch.no_grad(): seg_A = seg(data["A"].cuda())[0]
        actions, dists, _ = policy(data["A"].cuda(), data["A_Lab"].cuda(), seg_A)
        total_dists = torch.cat([total_dists, dists.probs], dim=0)

        with torch.no_grad():
            for j in range(data["A"].size(0)):
                A = data["A"][j:j+1]; A_gray = data["A_gray"][j:j+1]
                score0 = get_score(A, data["mask"][j])
                niqe0 = get_niqe(A)
                gan.set_input_A(A, A_gray)
                gan.set_input_A_origin(A)
                if actions[j] > 0:
                    for _ in range(actions[j]):
                        # GAN prediction ###########################
                        _, fake_B_tensor = gan.predict()
                        # GAN reset image data #########################
                        A = fake_B_tensor.clamp(-1, 1)
                        r,g,b = A[0:1, 0:1]+1, A[0:1, 1:2]+1, A[0:1, 2:3]+1
                        A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2. # h, w
                        gan.set_input_A(A, A_gray)
                    # Seg & NIQE reward ######################################
                    score1 = get_score(A, data["mask"][j])
                    # Reward ####################################
                    # reward = get_reward([score0, score1])
                    # reward = score1 - score0
                    mIoU_gain.append(score1 - score0)
                    niqe1 = get_niqe(A)
                    niqe_gain.append(niqe1 - niqe0)
                    reward = np.clip(score1 - score0, -r_neg, r_pos)
                    # reward_history[0] = reward_history[0] * reward_history[1] + reward * (1 - reward_history[1])
                    rewards[j] = gamma * (reward - reward_history[0])
                else:
                    _, fake_B_tensor = gan.predict()
                    # GAN reset image data #########################
                    A = fake_B_tensor.clamp(-1, 1)
                    # Seg & NIQE reward ######################################
                    score1 = get_score(A, data["mask"][j])
                    # Reward ####################################
                    # reward = get_reward([score1, score0])
                    # reward = score0 - score1
                    mIoU_gain.append(0)
                    niqe_gain.append(0)
                    reward = np.clip(score0 - score1, -r_pos, r_neg)
                    # reward_history[0] = reward_history[0] * reward_history[1] + reward * (1 - reward_history[1])
                    rewards[j] = gamma * (reward - reward_history[0])
                tbar.set_description('policy: %s, action: %s, reward: %.1f, reward_history: %.1f' % (str(torch.round(dists.probs.data[j] * 10 ** 2) / 10 ** 2), str(actions.data[j]), rewards[j], reward_history[0]))

        # back-propagate the hybrid loss
        loss = 0
        for idx in range(len(rewards)):
            loss -= (lambd_entropy * dists.entropy()[idx])
            loss += (- dists.log_prob(actions[idx])[idx] * rewards[idx])
            if rewards[idx] > 0: P += 1
            elif rewards[idx] <= 0: N += 1
            else: pass
            # print(Dists[idx].entropy(), - Dists[idx].log_prob(Actions[idx]) * Rewards[idx])
        torch.autograd.backward(loss)

        total_loss += loss.data.detach().cpu().numpy()
        if i > 0 and (i + 1) % update_interval == 0:
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalars('Loss', {'loss': loss.data.detach().cpu().numpy()}, epoch * len(tbar) + i)
            writer.add_scalars('Policy', {'P': 1. * P / (P + N), 'N': 1. * N / (P + N)}, epoch * len(tbar) + i)
            writer.add_scalars('mIoU', {"mIoU_gain": np.mean(mIoU_gain)}, epoch * len(tbar) + i)
            writer.add_scalars('NIQE', {"NIQE_gain": np.mean(niqe_gain)}, epoch * len(tbar) + i)
            mIoU_gain = []; niqe_gain = []
            total_loss = 0; P = N = 0

        if i > 0 and (i + 1) % update_reward_interval == 0:
            infer_actions = total_dists.argmax(1)
            mode_action = infer_actions.mode()[0]
            percent = (infer_actions == mode_action).sum().float() / len(infer_actions)
            print("policy collapse:", np.round(percent, 3), mode_action)
            print("r_pos:", r_pos, "r_neg:", r_neg, "==>")
            if percent >= 0.55:
                delta = (percent - 0.55) / 2.
                # if mode_action > 0: r_neg += 0.1; r_pos -= 0.1; r_pos = max(r_pos, 0)
                # else: r_pos += 0.1; r_neg -= 0.1; r_neg = max(r_neg, 0)
                if mode_action > 0: r_neg += delta; r_pos -= delta; r_pos = max(r_pos, 0.1)
                else: r_pos += delta; r_neg -= delta; r_neg = max(r_neg, 0.1)
            print("r_pos:", r_pos, "r_neg:", r_neg)
            total_dists = torch.empty(0, 4).cuda()


        # img_path = gan.get_image_paths()
        # print('process image... %s' % img_path)
        # visualizer.save_images(webpage, visuals, img_path)

    utils_seg.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': policy.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'best_pred': self.best_pred,
    }, opt_seg, True)
    return reward_history, r_neg, r_pos


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
                inter, union = get_mIoU(data["A"], data["mask"], inter_union=True)
                total_inter += inter; total_union += union
                niqe = get_niqe(data["A"])
                total_niqe.append(niqe)
                idx = total_union > 0
                tbar.set_description('mIoU: %.3f, NIQE: %.3f' % ((1.0 * total_inter[idx] / (np.spacing(1) + total_union[idx])).mean(), np.mean(total_niqe)))
            elif mode == "gan":
                inter, union = get_mIoU(data["A"], data["mask"], inter_union=True)
                total_inter_origin += inter; total_union_origin += union
                niqe = get_niqe(data["A"])
                total_niqe_origin.append(niqe)

                gan.set_input(data)
                visuals, fake_B_tensor = gan.predict(seg)
                inter, union = get_mIoU(fake_B_tensor, mask, inter_union=True)
                total_inter += inter; total_union += union
                idx = total_union > 0
                niqe = get_niqe(fake_B_tensor)
                total_niqe.append(niqe)
                tbar.set_description('mIoU_ori: %.3f, mIoU: %.3f, NIQE_ori: %.3f, NIQE: %.3f' % ((\
                    1.0 * total_inter_origin[idx] / (np.spacing(1) + total_union_origin[idx])).mean(),\
                    (1.0 * total_inter[idx] / (np.spacing(1) + total_union[idx])).mean(),\
                    np.mean(total_niqe_origin), np.mean(total_niqe)))
            elif mode == "patch":
                image = data['A']; image = image.cuda()
                mask = data['mask']

                inter, union = get_mIoU(image, mask, inter_union=True)
                total_inter_origin += inter; total_union_origin += union
                idx_origin = total_union_origin > 0
                niqe = get_niqe(image)
                total_niqe_origin.append(niqe)

                crop_size = opt_gan.fineSize
                _, _, h, w = image.size()
                n_h = int(np.ceil(1. * h / crop_size))
                n_w = int(np.ceil(1. * w / crop_size))
                step_h = 1.0 * (h - crop_size) / (n_h - 1) if n_h > 1 else 0
                step_w = 1.0 * (w - crop_size) / (n_w - 1) if n_w > 1 else 0
                image_new = torch.zeros(1, 3, h, w).cuda()
                seg_pred = torch.zeros(1, 19, h, w).cuda()
                template = torch.zeros(1, 3, h, w)

                for i in range(n_w):
                    for j in range(n_h):
                        # x is w, y is h
                        x = int(np.round(i * step_w)); y = int(np.round(j * step_h))
                        A = image[0:1, :, y: y + crop_size, x: x + crop_size]
                        r,g,b = A[0, 0, :, :]+1, A[0, 1, :, :]+1, A[0, 2, :, :]+1
                        A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2. # h, w
                        A_gray = A_gray.unsqueeze(0).unsqueeze(0)

                        luminance = np.sort((255 - A_gray*255).detach().cpu().numpy().flatten())
                        length = luminance.shape[0]
                        luminance = luminance[int(np.round(length * 0.1)) : int(np.round(length * 0.9))].mean()
                        if luminance < 60: N = 1
                        elif luminance < 80: N = 1
                        elif luminance < 200: N = 0
                        else : N = 0

                        # GAN prediction ###########################
                        if N == 0:
                            pred = F.softmax(seg(A)[0], dim=1) + F.softmax(seg(A.flip(3))[0], dim=1).flip(3)
                        else:
                            gan.set_input_A_origin(A)
                            for _ in range(N):
                                # GAN prediction ###########################
                                gan.set_input_A(A, A_gray)#, A_edge)
                                visuals, A = gan.predict(seg)
                                # GAN reset image data #########################
                                A = A.clamp(-1, 1)
                                pred = F.softmax(seg(A)[0], dim=1) + F.softmax(seg(A.flip(3))[0], dim=1).flip(3)
                                r,g,b = A[0, 0, :, :]+1, A[0, 1, :, :]+1, A[0, 2, :, :]+1
                                A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2. # h, w
                                A_gray = A_gray.unsqueeze(0).unsqueeze(0)
                        image_new[0:1, :, y: y + crop_size, x: x + crop_size] += A
                        seg_pred[0:1, :, y: y + crop_size, x: x + crop_size] += pred
                        template[0:1, :, y: y + crop_size, x: x + crop_size] += 1
                image_new = image_new.detach().cpu()
                image_new /= template
                # evaluation ############################
                inter, union = get_mIoU_from_softmax(seg_pred, mask, inter_union=True)
                total_inter += inter; total_union += union
                idx = total_union > 0
                niqe = get_niqe(image_new)
                total_niqe.append(niqe)
                tbar.set_description('mIoU_ori: %.3f, mIoU: %.3f, NIQE_ori: %.3f, NIQE: %.3f' % ((\
                    1.0 * total_inter_origin[idx_origin] / (np.spacing(1) + total_union_origin[idx_origin])).mean(),\
                    (1.0 * total_inter[idx] / (np.spacing(1) + total_union[idx])).mean(),\
                    np.mean(total_niqe_origin), np.mean(total_niqe)))
            elif mode == "policy":
                image = Image.open(data["A_paths"][0])
                mask = _mask_transform(Image.open(os.path.join("/ssd1/chenwy/bdd100k/seg/labels/", "val", os.path.splitext(data["A_paths"][0].split("/")[-1])[0] + '_train_id.png')))

                inter, union = get_mIoU(transformer(image).unsqueeze(0), mask, inter_union=True)
                total_inter_origin += inter; total_union_origin += union
                niqe = get_niqe(transformer(image).unsqueeze(0))
                total_niqe_origin.append(niqe)

                crop_size = opt_gan.fineSize
                # _, _, h, w = image.size()
                w, h = image.size
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
                        A_img = image.crop((x, y, x+crop_size, y+crop_size))
                        A = transformer(A_img).unsqueeze(0).cuda()
                        r,g,b = A[0, 0, :, :]+1, A[0, 1, :, :]+1, A[0, 2, :, :]+1
                        A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2. # h, w
                        A_gray = A_gray.unsqueeze(0).unsqueeze(0).cuda()
                        A_Lab = torch.Tensor(color.rgb2lab(np.array(A_img)) / 100).permute([2, 0, 1]).unsqueeze(0).cuda()
                        # A_border = transformer(image.crop((x-crop_size//2, y-crop_size//2, x+2*crop_size, y+2*crop_size))).unsqueeze(0).cuda()
                        # r,g,b = A_border[0, 0, :, :]+1, A_border[0, 1, :, :]+1, A_border[0, 2, :, :]+1
                        # A_gray_border = 1. - (0.299*r+0.587*g+0.114*b)/2. # h, w
                        # A_gray_border = A_gray_border.unsqueeze(0).unsqueeze(0).cuda()

                        # action is a list of prob. ##########################
                        # input = A_border.cuda()
                        # input = torch.cat([A_border, A_gray_border], dim=1).cuda()
                        # input = torch.cat([A_border.cuda(), A_gray_border.cuda(), seg(A_border.cuda())[0]], dim=1)
                        # actions, dists = policy(input)
                        with torch.no_grad(): seg_A = seg(A)[0]
                        actions, dists = policy(A, A_Lab, seg_A)
                        # total_dists = torch.cat([total_dists, dists.probs], dim=0)

                        with torch.no_grad():
                            for j in range(A.size(0)):
                                gan.set_input_A(A, A_gray)
                                gan.set_input_A_origin(A)
                                if actions[j] > 0:
                                    for _ in range(actions[j]):
                                        # GAN prediction ###########################
                                        _, fake_B_tensor = gan.predict()
                                        # GAN reset image data #########################
                                        A = fake_B_tensor.clamp(-1, 1)
                                        r,g,b = A[0:1, 0:1]+1, A[0:1, 1:2]+1, A[0:1, 2:3]+1
                                        A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2. # h, w
                                        gan.set_input_A(A, A_gray)

                        total_n.append(int(actions[j].detach().cpu().numpy()))
                        image_new[0:1, :, y: y + crop_size, x: x + crop_size] += A.detach().cpu()
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


update_interval = 3
update_reward_interval = 20
epochs = 800
reward_history = [0, 0.975] # mean, smooth_ratio

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if evaluation:
    evaluate(mode=evaluation) # "gan"
else:
    # optimizer = torch.optim.SGD(policy.parameters(), lr, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(params_list, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    opt_seg.dataset = "bdd100k_seg"
    opt_seg.model = "policy"
    opt_seg.checkname = "image.Lab.seg_0.75_msn.act4_adapt20.0.55.clip5.5_entropy0.3_lr3e3.800epoch_" + time.strftime("%Y-%m-%d-%H-%M",time.localtime())
    print(opt_seg.checkname)
    writer = SummaryWriter(log_dir=os.path.join("/home/chenwy/DynamicLightEnlighten/logs", opt_seg.checkname, time.strftime("%Y-%m-%d-%H-%M",time.localtime())))

    for epoch in range(epochs):
        print(opt_seg.checkname)
        reward_history, r_neg, r_pos = train(epoch, reward_history, r_neg, r_pos)
        # evaluate()

webpage.save()
#####################################################
