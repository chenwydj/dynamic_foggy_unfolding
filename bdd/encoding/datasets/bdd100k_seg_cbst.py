###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import os
import sys
import numpy as np
import random
import math
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as transform
import re
from tqdm import tqdm
from .base import BaseDataset
import json
import heapq

class BDD100K_Seg(BaseDataset):
    BASE_DIR = 'bdd100k'
    NUM_CLASS = 19
    trigger = None # 'A', 'B', 'AB'
    topk = []; heapq.heapify(topk)
    portion = 0.2

    def __init__(self, root='/ssd1/chenwy/', split='train', mode=None, transform=None, target_transform=None, **kwargs):
        super(BDD100K_Seg, self).__init__(root, split, mode, transform, target_transform, **kwargs)

        self.class_freq = np.ones(self.NUM_CLASS) * 1./self.NUM_CLASS
        # assert exists
        self.root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(self.root), "Please download the dataset!!"

        # load json label file #################
        f = open(os.path.join(self.root, "labels/bdd100k_labels_images_train.json"))
        self.train_labels = json.load(f)
        self.name2index_train = {}
        for i in range(len(self.train_labels)):
            self.name2index_train[self.train_labels[i]['name']] = i

        f = open(os.path.join(self.root, "labels/bdd100k_labels_images_val.json"))
        self.val_labels = json.load(f)
        self.name2index_val = {}
        for i in range(len(self.val_labels)):
            self.name2index_val[self.val_labels[i]['name']] = i
        ########################################

        self.images_A, self.masks_A = _get_bdd100k_seg_pairs(os.path.join(self.root, 'seg_luminance/100_255'), os.path.join(self.root, 'seg/labels'), split)
        if split == 'train':
            self.images_B, self.masks_B = _get_bdd100k_seg_pairs(os.path.join(self.root, 'seg_luminance/0_75'), os.path.join(self.root, 'seg_luminance/0_75_cbst/labels'), split)
        else:
            self.images_B, self.masks_B = _get_bdd100k_seg_pairs(os.path.join(self.root, 'seg_luminance/0_75'), os.path.join(self.root, 'seg/labels'), split)
        self.total_pixels = 720 * 1280 * len(self.images_B)
        self.k = int(self.portion * self.total_pixels)
        if split != 'vis':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + self.root + "\n"))

    def __getitem__(self, index):
        if self.trigger == 'A' or (self.trigger == 'AB' and index < len(self.images_A)):
            img = Image.open(self.images_A[index]).convert('RGB')
            mask = Image.open(self.masks_A[index])
            name = self.images_A[index].split('/')[-1]
            # synchrosized transform
            if self.mode == 'train':
                img, mask, vector = self._sync_transform(img, mask, class_freq=self.class_freq)
                self.class_freq = 0.99 * self.class_freq + 0.01 * vector
            elif self.mode == 'val':
                img, mask = self._val_sync_transform(img, mask)
        else:
            if self.trigger == 'AB' and index >= len(self.images_A): index = index - len(self.images_A)
            img = Image.open(self.images_B[index]).convert('RGB')
            mask = Image.open(self.masks_B[index])
            name = self.images_B[index].split('/')[-1]
            if self.mode == 'train':
                img, mask = self._sync_transform(img, mask)
            elif self.mode == 'val':
                mask = Image.open(self.masks_B[index])
                img, mask = self._val_sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and mask is not None:
            mask = self.target_transform(mask)

        return img, mask, self.images[index], self.class_freq

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        if self.trigger == 'AB': return len(self.images_A) + len(self.images_B)
        elif self.trigger == "A": return len(self.images_A)
        elif self.trigger == "B": return len(self.images_B)
        else: return 0

    @property
    def pred_offset(self):
        return 0
    
    def _update_topk_queue(self, values, name):
        values = values.detach().cpu().numpy()[0]
        self._save_pred(values, name)
        values = values.max(axis=0).reshape(-1).tolist()
        for v in values:
            if len(self.topk) < self.k or v > self.topk[0]: heapq.heappush(self.topk, v)
            while len(self.topk) > self.k:
                heapq.heappop(self.topk)
    
    def _save_pred(self, value, name):
        np.save(os.path.join(self.root, "seg_luminance/0_75_cbst/softmax", name) + ".npy", value)

    def _update_mask_B(self):
        for name in os.listdir(os.path.join(self.root, "seg_luminance/0_75_cbst/softmax")):
            probs = np.load(os.path.join(self.root, "seg_luminance/0_75_cbst/softmax", name))
            mask = probs.max(axis=0) >= self.topk[0]
            labels = probs.argmax(axis=0)
            labels[mask] = 255
            Image.fromarray(labels).convert("L").save(os.path.join(self.root, "seg_luminance/0_75_cbst/labels/train", name[:-4] + ".png"))
        return
    
    def _set_trigger(self, trigger):
        # trigger = 'A'|B'|'AB'
        self.trigger = trigger

    def _set_portion(self, portion):
        self.portion = portion
        self.k = int(self.portion * self.total_pixels)


def _get_bdd100k_seg_pairs(img_folder, mask_folder, split='train'):
    def get_path(img_folder):
        img_paths = []
        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                if os.path.isfile(imgpath):
                    img_paths.append(imgpath)
                else:
                    print('cannot find the image:', imgpath)
        return img_paths
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                maskname = basename + '_train_id.png'
                maskpath = os.path.join(mask_folder, maskname)
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths

    img_folder = os.path.join(img_folder, split)
    if split == 'train' or split == 'val':
        mask_folder = os.path.join(mask_folder, split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    else: # "test"
        img_paths = get_path(img_folder)

    return img_paths, mask_paths
