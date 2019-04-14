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

class Mapillary(BaseDataset):
    BASE_DIR = 'mapillary'
    NUM_CLASS = 66
    def __init__(self, root='/ssd1/chenwy/', split='train', mode=None, transform=None, target_transform=None, **kwargs):
        super(Mapillary, self).__init__(root, split, mode, transform, target_transform, **kwargs)
        self.class_freq = np.ones(self.NUM_CLASS) * 1./self.NUM_CLASS
        # assert exists
        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Please download the dataset!!"

        self.images, self.masks = _get_mapillary_seg_pairs(root, split)
        if split != 'vis':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'vis':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        
        mask = Image.open(self.masks[index])
        
        # synchrosized transform
        if self.mode == 'train':
            img, mask, vector = self._sync_transform(img, mask, class_freq=self.class_freq)
            self.class_freq = 0.99 * self.class_freq + 0.01 * vector
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        # name = self.images[index].split('/')[-1]
        return img, mask, self.images[index], self.class_freq

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_mapillary_seg_pairs(folder, split='train'):
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
                maskname = basename + '.png'
                maskpath = os.path.join(mask_folder, maskname)
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths

    if split == 'train':
        # img_folder = os.path.join(folder, 'training/images')
        # img_folder = os.path.join(folder, 'seg_enhanced/images/train')
        # img_folder = os.path.join(folder, 'seg_daytime/images/train')
        # img_folder = os.path.join(folder, 'light_enhance_AB/seg_85/trainB')
        img_folder = os.path.join(folder, 'luminance/100_255/training/images')
        mask_folder = os.path.join(folder, 'training/labels')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    elif split == 'val':
        # img_folder = os.path.join(folder, 'validation/images')
        # img_folder = os.path.join(folder, 'seg_enhanced/images/val')
        # img_folder = os.path.join(folder, 'seg_daytime/images/val')
        # img_folder = os.path.join(folder, 'light_enhance_AB/seg_85/testB')
        img_folder = os.path.join(folder, 'luminance/100_255/validation/images')
        mask_folder = os.path.join(folder, 'validation/labels')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    elif split == 'test':
        img_folder = os.path.join(folder, 'seg/images/test')
        img_paths = get_path(img_folder)
    else:
        train_img_folder = os.path.join(folder, 'seg/images/train')
        # train_img_folder = os.path.join(folder, 'seg_enhanced/images/train')
        train_mask_folder = os.path.join(folder, 'seg/labels/train')
        val_img_folder = os.path.join(folder, 'seg/images/val')
        # val_img_folder = os.path.join(folder, 'seg_enhanced/images/val')
        val_mask_folder = os.path.join(folder, 'seg/labels/val')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        return train_img_paths + val_img_paths, train_mask_paths + val_mask_paths

    return img_paths, mask_paths
