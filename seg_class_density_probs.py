import numpy as np
from tqdm import tqdm
import os
import re
from PIL import Image
import random

w = 2048; h = 1024
probs = []
n = 10

mask_paths = []
root = "/ssd1/chenwy/cityscapes/"
with open(os.path.join(root, "train_fine.txt"), 'r') as lines:
    for line in tqdm(lines):
        ll_str = re.split(' ', line)
        maskpath = os.path.join(root, ll_str[1].rstrip())
        if os.path.isfile(maskpath):
            mask_paths.append(maskpath)
        else:
            print('cannot find the mask:', maskpath)

crop_size = 500
for name in tqdm(mask_paths):
    label = Image.open(os.path.join(root, name))
    for _ in range(n):
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        label_patch = np.array(label.crop((x1, y1, x1+crop_size, y1+crop_size))).astype('int32') # cropped mask for light_enhance_AB/seg
        unique, counts = np.unique(label_patch, return_counts=True)
        # calculate class density prob. vector
        unique[unique == 255] = -1; unique += 1; prob = np.zeros(20); prob[unique] = counts/counts.sum()
        probs.append(prob)

np.save("seg_class_density_probs_train_500.npy", probs)