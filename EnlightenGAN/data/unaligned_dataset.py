import torch
from torch import nn
import os.path
import os
import re
from tqdm import tqdm
import torchvision.transforms as transforms
from EnlightenGAN.data.base_dataset import BaseDataset, get_transform
from EnlightenGAN.data.image_folder import make_dataset
import random
from PIL import Image
import PIL
from pdb import set_trace as st
import numpy as np
from skimage import color, feature
from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans


def pad_tensor(input):
    
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

            padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
            input = padding(input).data
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.shape[2], input.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:,:, pad_top: height - pad_bottom, pad_left: width - pad_right]


class UnalignedDataset(BaseDataset):

    def get_path_pairs(self, folder, split_f):
        img_paths = []
        mask_paths = []
        with open(split_f, 'r') as lines:
            for line in tqdm(lines):
                ll_str = re.split(' ', line)
                imgpath = os.path.join(folder,ll_str[0].rstrip())
                maskpath = os.path.join(folder,ll_str[1].rstrip())
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # self.class_freq = np.ones(19) * 1./19
        # self.class_weights = ((1 - self.class_freq) ** 5); self.class_weights /= self.class_weights.max()

        #######################################################
        # kmeans cluster on seg class density #######################
        seg_class_density = np.load("/ssd1/chenwy/cityscapes/seg_class_density_probs_train_500.npy")[:, 1:] # ignore index at 0 (-1 in label png)
        self.cluster = KMeans(n_clusters=5)
        self.cluster.fit(seg_class_density)

        if opt.phase == 'train':
            split_f = os.path.join(opt.dataroot, 'train_fine_foggy_0.01.txt')
            self.A_paths, self.mask_paths = self.get_path_pairs(opt.dataroot, split_f)
            split_f = os.path.join(opt.dataroot, 'train_fine.txt')
            self.B_paths, _ = self.get_path_pairs(opt.dataroot, split_f)
        elif opt.phase == 'val':
            split_f = os.path.join(opt.dataroot, 'val_fine_foggy_0.01.txt')
            self.A_paths, self.mask_paths = self.get_path_pairs(opt.dataroot, split_f)
            split_f = os.path.join(opt.dataroot, 'val_fine.txt')
            self.B_paths, _ = self.get_path_pairs(opt.dataroot, split_f)
        else:
            split_f = os.path.join(opt.dataroot, 'test_foggy_0.005.txt')
            self.A_paths, self.mask_paths = self.get_path_pairs(opt.dataroot, split_f)
            split_f = os.path.join(opt.dataroot, 'test.txt')
            self.B_paths, _ = self.get_path_pairs(opt.dataroot, split_f)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        self.transform = get_transform(opt)

    def __getitem__(self, index_A):
        A_path = self.A_paths[index_A % self.A_size]
        mask_path = self.mask_paths[index_A % self.A_size]
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B % self.B_size]
        # mask_B_path = self.mask_paths[index_B % self.B_size]

        A_image = Image.open(A_path).convert('RGB')
        mask = Image.open(mask_path)
        B_image = Image.open(B_path).convert('RGB')
        # mask_B = Image.open(mask_B_path)

        w, h = A_image.size
        if self.opt.phase == "train":
            short_size = random.randint(int(h*0.5), int(h*1.25))
            oh = short_size
            ow = int(1.0 * w * oh / h)
            A_image = A_image.resize((ow, oh), Image.BILINEAR)
            B_image = B_image.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)
            # mask_B = mask_B.resize((ow, oh), Image.NEAREST)
        w, h = A_image.size

        if self.opt.phase == "val":
            # whole image/mask #####################
            A_img = A_image
            B_img = B_image
            category_A = 1
            A_npy = np.array(A_img)
            # B_npy = np.array(B_img)

            # r,g,b = A_npy[:, :, 0], A_npy[:, :, 1], A_npy[:, :, 2]
            # value_A = (0.299*r+0.587*g+0.114*b) / 255.
            # value_A = np.sort(value_A.flatten())
            # length = value_A.shape[0]
            # value_A = value_A[int(np.round(length * 0.1)) : int(np.round(length * 0.9))].mean()

            mask = np.array(mask).astype('int32') # cropped mask for light_enhance_AB/seg
            mask = self._mask_transform(mask)
        
            # A_boundary = Image.open(os.path.join("/ssd1/chenwy/bdd100k/seg_luminance/0_100_boundary/", self.opt.phase, os.path.splitext(A_path.split("/")[-1])[0] + '.png'))
            # A_boundary = np.array(A_boundary).astype('float32')
            # A_boundary = torch.from_numpy(A_boundary).unsqueeze(0)
            # A_gt = Image.open(os.path.join("/ssd1/chenwy/bdd100k/seg_luminance/0_100_gt/", self.opt.phase, A_path.split("/")[-1])).convert('RGB')
            # A_gt = self.transform(A_gt)
            # A_boundary = torch.zeros(1)
            # A_gt = torch.zeros(1)
        else:
            ###################################################
            # patch selection ###########################
            x1 = random.randint(0, w - self.opt.fineSize)
            y1 = random.randint(0, h - self.opt.fineSize)
            x2 = x1 + random.randint(-self.opt.fineSize//2, self.opt.fineSize//2); x2 = max(0, x2); x2 = min(x2, w - self.opt.fineSize)
            y2 = y1 + random.randint(-self.opt.fineSize//2, self.opt.fineSize//2); y2 = max(0, y2); y2 = min(y2, w - self.opt.fineSize)
            A_img = A_image.crop((x1, y1, x1+self.opt.fineSize, y1+self.opt.fineSize))
            B_img = B_image.crop((x2, y2, x2+self.opt.fineSize, y2+self.opt.fineSize))
            A_npy = np.array(A_img)

            # r,g,b = A_npy[:, :, 0], A_npy[:, :, 1], A_npy[:, :, 2]
            # value_A = (0.299*r+0.587*g+0.114*b) / 255.
            # value_A = np.sort(value_A.flatten())
            # length = value_A.shape[0]
            # value_A = value_A[int(np.round(length * 0.15)) : int(np.round(length * 0.85))].mean()

            # B_npy = np.array(B_img)
            # r,g,b = B_npy[:, :, 0], B_npy[:, :, 1], B_npy[:, :, 2]
            # value_B = (0.299*r+0.587*g+0.114*b) / 255.
            # value_B = np.sort(value_B.flatten())
            # length = value_B.shape[0]
            # value_B = value_B[int(np.round(length * 0.15)) : int(np.round(length * 0.85))].mean()

            mask = np.array(mask.crop((x1, y1, x1+self.opt.fineSize, y1+self.opt.fineSize))).astype('int32') # cropped mask
            unique, counts = np.unique(mask, return_counts=True)
            # calculate class density prob. vector ####
            unique[unique == 255] = -1; unique += 1; prob_A = np.zeros(20); prob_A[unique] = counts/counts.sum()
            # if prob_A[0] > 0.35: continue # drop images with too much ignore (-1) areas
            prob_A = prob_A[1:]
            category_A = self.cluster.predict(prob_A.reshape(1, -1))
            # class-balanced selection ###
            # prob = self.class_weights.dot(prob_A)
            # if random.random() < prob: continue
            ##############################
            # select by class diversity
            # if len(unique) < 2 or (counts / counts.sum()).max() > 0.7: n_try += 1; continue
            mask = self._mask_transform(mask)

            # mask_B = np.array(mask_B.crop((x2, y2, x2+self.opt.fineSize, y2+self.opt.fineSize))).astype('int32') # cropped mask_B
            # unique, counts = np.unique(mask_B, return_counts=True)
            # calculate class density prob. vector ####
            # unique[unique == 255] = -1; unique += 1; prob_B = np.zeros(20); prob_B[unique] = counts/counts.sum(); prob_B = prob_B[1:]
            # compare & threshold class density prob. vector
            # if (((prob_A - prob_B) ** 2).mean()) ** 0.5 > 0.16: continue

            # self.class_freq = 0.99 * self.class_freq + 0.01 * prob_A
            # self.class_weights = ((1 - self.class_freq) ** 5); self.class_weights /= self.class_weights.max()

            # load A ground truth image without foggy ############
            # A_gt = Image.open(os.path.join("/ssd1/chenwy/bdd100k/seg_luminance/0_100_gt/", self.opt.phase, A_path.split("/")[-1])).convert('RGB')
            # A_gt = A_gt.resize((w, h), Image.BILINEAR)
            # A_gt = A_gt.crop((x1, y1, x1+self.opt.fineSize, y1+self.opt.fineSize))
            # A_gt = self.transform(A_gt)
            ##########################################################################

        A_Lab = torch.Tensor(color.rgb2lab(A_npy) / 100).permute([2, 0, 1])

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        if self.opt.resize_or_crop == 'no':
            r,g,b = A_img[0]+1, A_img[1]+1, A_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
            input_img = A_img
            # A_gray = (1./A_gray)/255.

            # r,g,b = A_img_border[0]+1, A_img_border[1]+1, A_img_border[2]+1
            # A_gray_border = 1. - (0.299*r+0.587*g+0.114*b)/2.
            # A_gray_border = torch.unsqueeze(A_gray_border, 0)
        else:
            w = A_img.size(2)
            h = A_img.size(1)
            
            # A_gray = (1./A_gray)/255.
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(2, idx)
                B_img = B_img.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(1, idx)
                B_img = B_img.index_select(1, idx)
            if self.opt.vary == 1 and (not self.opt.no_flip) and random.random() < 0.5:
                times = random.randint(self.opt.low_times,self.opt.high_times)/100.
                input_img = (A_img+1)/2./times
                input_img = input_img*2-1
            else:
                input_img = A_img
            if self.opt.lighten:
                B_img = (B_img + 1)/2.
                B_img = (B_img - torch.min(B_img))/(torch.max(B_img) - torch.min(B_img))
                B_img = B_img*2. -1
            r,g,b = input_img[0]+1, input_img[1]+1, input_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
        return {'A': A_img, 'B': B_img, 'A_gray': A_gray, 'input_img': input_img,
                'A_paths': A_path, 'B_paths': B_path, 'mask': mask,
                'A_Lab': A_Lab,
                # 'A_gt': A_gt,
                'category': category_A, 
                }

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()
