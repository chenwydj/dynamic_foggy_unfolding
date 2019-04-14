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

class BDD100K_Drivable(BaseDataset):
    BASE_DIR = 'bdd100k'
    NUM_CLASS = 3
    weather2num = {"clear": 0, "foggy": 1, "overcast": 2, "partly cloudy": 3, "rainy": 4, "snowy": 5, "undefined": 6, }
    timeofday2num = {"dawn/dusk": 0, "daytime": 1, "night": 2, "undefined": 3}
    scene2num = {"city street": 0, "gas stations": 1, "highway": 2, "parking lot": 3, "residential": 4, "tunnel": 5, "undefined": 6, }
    def __init__(self, root='/ssd1/chenwy/', split='train', mode=None, transform=None, target_transform=None, **kwargs):
        super(BDD100K_Drivable, self).__init__(root, split, mode, transform, target_transform, **kwargs)
        # assert exists
        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Please download the dataset!!"

        # load json label file #################
        f = open(os.path.join(root, "labels/bdd100k_labels_images_train.json"))
        self.train_labels = json.load(f)
        self.name2index_train = {}
        for i in range(len(self.train_labels)):
            self.name2index_train[self.train_labels[i]['name']] = i

        f = open(os.path.join(root, "labels/bdd100k_labels_images_val.json"))
        self.val_labels = json.load(f)
        self.name2index_val = {}
        for i in range(len(self.val_labels)):
            self.name2index_val[self.val_labels[i]['name']] = i
        ########################################

        self.images, self.masks = _get_bdd100k_drivable_pairs(root, split)
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
            img, mask = self._sync_transform(img, mask)
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

        name = self.images[index].split('/')[-1]
        # get weather, timeofday, scene ################
        # if name in self.name2index_train:
        #     attributes = self.train_labels[self.name2index_train[name]]['attributes']
        # elif name in self.name2index_val:
        #     attributes = self.val_labels[self.name2index_val[name]]['attributes']
        # else:
        #     attributes = {'weather': 'undefined', 'timeofday': 'undefined', 'scene': 'undefined'}
        # weather = torch.from_numpy(np.array(self.weather2num[attributes['weather']]).reshape(1).astype('int32')).long()
        # timeofday = torch.from_numpy(np.array(self.timeofday2num[attributes['timeofday']]).reshape(1).astype('int32')).long()
        # scene = torch.from_numpy(np.array(self.scene2num[attributes['scene']]).reshape(1).astype('int32')).long()
        ################################################

        # return img, mask, weather, timeofday, scene, self.images[index]
        return img, mask, self.images[index]

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_bdd100k_drivable_pairs(folder, split='train'):
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
                maskname = basename + '_drivable_id.png'
                maskpath = os.path.join(mask_folder, maskname)
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths

    if split == 'train':
        # img_folder = os.path.join(folder, 'images/100k/train')
        img_folder = os.path.join(folder, 'images_enhanced/100k/train')
        mask_folder = os.path.join(folder, 'drivable_maps/labels/train')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    elif split == 'val':
        # img_folder = os.path.join(folder, 'images/100k/val')
        img_folder = os.path.join(folder, 'images_enhanced/100k/val')
        mask_folder = os.path.join(folder, 'drivable_maps/labels/val')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    elif split == 'test':
        img_folder = os.path.join(folder, 'images/100k/test')
        img_paths = get_path(img_folder)
    else:
        # train_img_folder = os.path.join(folder, 'images/100k/train')
        train_img_folder = os.path.join(folder, 'images_enhanced/100k/train')
        train_mask_folder = os.path.join(folder, 'drivable_maps/labels/train')
        # val_img_folder = os.path.join(folder, 'images/100k/val')
        val_img_folder = os.path.join(folder, 'images_enhanced/100k/val')
        val_mask_folder = os.path.join(folder, 'drivable_maps/labels/val')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        return train_img_paths + val_img_paths, train_mask_paths + val_mask_paths

    return img_paths, mask_paths
