###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import torch.utils.data as data

__all__ = ['BaseDataset', 'test_batchify_fn']

class BaseDataset(data.Dataset):
    def __init__(self, root, split, mode=None, transform=None, 
                 target_transform=None, base_size=520, crop_size=480,
                 logger=None, scale=True):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        self.logger = logger
        self.scale = scale

        if self.mode == 'train':
            print('BaseDataset: base_size {}, crop_size {}'. \
                format(base_size, crop_size))

        if not self.scale:
            if self.logger is not None:
                self.logger.info('single scale training!!!')


    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplemented

    def _val_sync_transform(self, img, mask=None):
        w, h = img.size
        short_size = self.base_size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        if mask is not None: mask = mask.resize((ow, oh), Image.NEAREST)

        # # center crop
        # w, h = img.size
        # x1 = int(round((w - outsize) / 2.))
        # y1 = int(round((h - outsize) / 2.))
        # img = img.crop((x1, y1, x1+outsize, y1+outsize))
        # mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
        # # final transform
        if mask is not None: return img, self._mask_transform(mask)
        else: return img

    def _sync_transform(self, image, mask=None, class_freq=None):
        # random mirror
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if mask is not None: mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # random scale (short edge from 480 to 720)
        if self.scale:
            short_size = random.randint(int(self.base_size*0.75), int(self.base_size*1.5))
            # short_size = random.randint(int(self.base_size*0.8), int(self.base_size*1.2))
        else:
            short_size = self.base_size
        w, h = image.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        image = image.resize((ow, oh), Image.BILINEAR)
        if mask is not None: mask = mask.resize((ow, oh), Image.NEAREST)

        # random rotate -10~10, mask using NN rotate
        # deg = random.uniform(-10, 10)
        # image = image.rotate(deg, resample=Image.BILINEAR)
        # mask = mask.rotate(deg, resample=Image.NEAREST)

        # pad crop
        crop_size = self.crop_size
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            image = ImageOps.expand(image, border=(0, 0, padw, padh), fill=0)
            if mask is not None: mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)#pad 255 for cityscapes

        # random crop crop_size
        w, h = image.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        image_crop = image.crop((x1, y1, x1+crop_size, y1+crop_size))
        if mask is not None: mask_crop = np.array(mask.crop((x1, y1, x1+crop_size, y1+crop_size)))
        if class_freq is not None and mask is not None:
            class_weights = ((1 - class_freq) ** 5); class_weights /= class_weights.max()
            while True:
                unique, counts = np.unique(mask_crop, return_counts=True)
                # calculate class density prob. vector
                unique[unique == 255] = -1; unique += 1; vector = np.zeros(len(class_freq)+1); vector[unique] = counts/counts.sum(); vector = vector[1:]
                prob = class_weights.dot(vector)
                if random.random() < prob: break
                x1 = random.randint(0, w - crop_size)
                y1 = random.randint(0, h - crop_size)
                image_crop = image.crop((x1, y1, x1+crop_size, y1+crop_size))
                mask_crop = np.array(mask.crop((x1, y1, x1+crop_size, y1+crop_size)))

        # gaussian blur as in PSP
        # if random.random() < 0.5:
        #     img = img.filter(ImageFilter.GaussianBlur(
        #         radius=random.random()))

        # final transform
        if class_freq is None:
            if mask is not None: return image_crop, self._mask_transform(mask_crop)
            else: return image_crop
        else: return image_crop, self._mask_transform(mask_crop), vector

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(np.array(mask)).long()


def test_batchify_fn(data):
    error_msg = "batch must contain tensors, tuples or lists; found {}"
    if isinstance(data[0], (str, torch.Tensor)):
        return list(data)
    elif isinstance(data[0], (tuple, list)):
        data = zip(*data)
        return [test_batchify_fn(i) for i in data]
    raise TypeError((error_msg.format(type(batch[0]))))
