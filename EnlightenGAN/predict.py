import time
import os
import torch
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from psnr import test_psnr
import torchvision.transforms as transforms

from util.util import tensor2im, save_image
from tqdm import tqdm


##### BDD Seg. #################################################
import bdd.encoding.utils as utils_seg
from bdd.encoding.parallel import DataParallelModel
from bdd.encoding.models import get_segmentation_model
from bdd.experiments.segmentation.option import Options

opt_seg = Options()#.parse()
opt_seg.dataset = "bdd100k_seg"
opt_seg.model = "fcn"
opt_seg.backbone = "resnet50"
opt_seg.dilated = True
opt_seg.ft = True
# opt_seg.ft_resume = "/home/chenwy/DynamicLightEnlighten/bdd/bdd100k_seg/fcn_model/res50_di_360px_daytime/model_best.pth.tar"
opt_seg.ft_resume = "/home/chenwy/DynamicLightEnlighten/bdd/bdd100k_seg/fcn_model/res50_di_180px_L100.255/model_best.pth.tar"
opt_seg.eval = True

seg = get_segmentation_model(opt_seg.model, dataset=opt_seg.dataset, backbone=opt_seg.backbone, aux=False, se_loss=False,
                               dilated=opt_seg.dilated,
                               # norm_layer=BatchNorm2d, # for multi-gpu
                               base_size=720, crop_size=180, multi_grid=False, multi_dilation=False)
seg = DataParallelModel(seg).cuda()
seg.eval()
if opt_seg.ft:
    checkpoint = torch.load(opt_seg.ft_resume)
    seg.module.load_state_dict(checkpoint['state_dict'], strict=False)
    # self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.ft_resume, checkpoint['epoch']))
####################################################



opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join("./ablation/", opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
print(len(dataset))

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

with torch.no_grad():
    for i, data in enumerate(tqdm(dataset)):
        model.set_input(data)
        visuals, _ = model.predict(seg)
        img_path = model.get_image_paths()
        # print('process image... %s' % img_path)
        save_image(visuals['fake_B'], "/ssd1/chenwy/bdd100k/seg_luminance/0_100_r1/train/" + img_path[0].split('/')[-1])
        # visualizer.save_images(webpage, visuals, img_path)

webpage.save()