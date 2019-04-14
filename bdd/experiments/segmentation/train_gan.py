###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import os
import copy
import numpy as np
from tqdm import tqdm
import time

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
from torch.nn import functional as F

import bdd.encoding.utils as utils
from bdd.encoding.nn import SegmentationLosses, BatchNorm2d
from bdd.encoding.nn import SegmentationMultiLosses
from bdd.encoding.parallel import DataParallelModel, DataParallelCriterion
from bdd.encoding.datasets import get_segmentation_dataset
from bdd.encoding.models import get_segmentation_model
from tensorboardX import SummaryWriter
from option import Options

import json


torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable



##### EnlightenGAN ################################################
from EnlightenGAN.options.test_options import TestOptions
from EnlightenGAN.data.data_loader import CreateDataLoader
from EnlightenGAN.models.models import create_model
from EnlightenGAN.psnr import test_psnr
import EnlightenGAN.util.util as utils_gan
from EnlightenGAN.util.util import tensor2im, save_image
from EnlightenGAN.util.visualizer import Visualizer
from EnlightenGAN.util import html

opt_gan = TestOptions()
opt_gan.gpu_ids = '0'
opt_gan.isTrain = False
opt_gan.input_nc = 3
opt_gan.output_nc = 3
opt_gan.vgg = 0
opt_gan.fcn = 0
opt_gan.checkpoints_dir = '/home/chenwy/DynamicLightEnlighten/EnlightenGAN/checkpoints/'
opt_gan.batchSize = 1
opt_gan.name = "bdd.seg3_gt.weight.0.1l1_day120.255.night0.65.classconstraint_G.multitask.seg.fcn_D.multi.layer3.2_vgg0_270px_align"
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

gan = create_model(opt_gan)
######################################################



class Trainer():
    def __init__(self, args):
        self.args = args
        args.log_name = str(args.checkname)
        self.logger = utils.create_logger(args.log_root, args.log_name)
        self.writer = SummaryWriter(log_dir=os.path.join(args.log_root, args.log_name, time.strftime("%Y-%m-%d-%H-%M",time.localtime())))
        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),
            # transform.Normalize([.485, .456, .406], [.229, .224, .225])
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        # dataset
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size, 'logger': self.logger, 'scale': args.scale}
        trainset = get_segmentation_dataset(args.dataset, split='train', mode='train', **data_kwargs)
        testset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)
        # self.valloader = data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=1, drop_last=False, shuffle=False, **kwargs)
        self.nclass = trainset.num_class

        # model
        model = get_segmentation_model(args.model, dataset=args.dataset, backbone=args.backbone, aux=args.aux, se_loss=args.se_loss,
                                       dilated=args.dilated,
                                       # norm_layer=BatchNorm2d, # for multi-gpu
                                       base_size=args.base_size, crop_size=args.crop_size, multi_grid=args.multi_grid, multi_dilation=args.multi_dilation)

        #####################################################################
        self.logger.info(model)
        # optimizer using different LR
        params_list = [{'params': model.pretrained.parameters(), 'lr': 1 * args.lr},]
        if hasattr(model, 'head'):
            params_list.append({'params': model.head.parameters(), 'lr': 1 * args.lr*10})
        if hasattr(model, 'auxlayer'):
            params_list.append({'params': model.auxlayer.parameters(), 'lr': 1 * args.lr*10})
        optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        if args.model == 'danet':
            self.criterion = SegmentationMultiLosses(nclass=self.nclass)
        elif args.model == 'fcn':
            self.criterion = SegmentationLosses(se_loss=args.se_loss, aux=args.aux, nclass=self.nclass)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        #####################################################################

        self.model, self.optimizer = model, optimizer
        # using cuda
        if args.cuda:
            self.model = DataParallelModel(self.model).cuda()
            self.criterion = DataParallelCriterion(self.criterion).cuda()
        # finetune from a trained model
        if args.ft:
            args.start_epoch = 0
            checkpoint = torch.load(args.ft_resume)
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.ft_resume, checkpoint['epoch']))
        # resuming checkpoint
        if args.resume:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        # lr scheduler
        self.scheduler = utils.LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.trainloader), logger=self.logger, lr_step=args.lr_step)
        self.best_pred = 0.0

        self.logger.info(self.args)

    def training(self, epoch):
        train_loss = 0.0

        ################################################
        self.model.train()
        ################################################

        tbar = tqdm(self.trainloader)

        self.optimizer.zero_grad()
        for i, (image, target, name, class_freq) in enumerate(tbar):
            ################################################
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            ################################################
            if torch_ver == "0.3":
                image = Variable(image)
                target = Variable(target)
            r,g,b = image[:, 0, :, :]+1, image[:, 1, :, :]+1, image[:, 2, :, :]+1
            gray = 1. - (0.299*r+0.587*g+0.114*b)/2. # h, w
            gray = gray.unsqueeze(1)
            with torch.no_grad(): fake_B, _, _ = gan.netG_A.forward(image, gray)
            outputs = self.model(fake_B.clamp(-1, 1))

            loss = self.criterion(outputs, target)

            loss.backward()
            if epoch % self.args.late_update == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f; ' % (train_loss / (i + 1)) + '[' + ' '.join([ "%.2f"%p for p in np.round(class_freq[0], 2)]) + ']')
        self.logger.info('Train loss: %.3f' % (train_loss / (i + 1)))

        # save checkpoint every 5 epoch
        #  is_best = False
        #  if epoch % 5 == 0:
        #      # filename = "checkpoint_%s.pth.tar"%(epoch+1)
        #      filename = "checkpoint_%s.%s.%s.%s.pth.tar"%(self.args.log_root, self.args.checkname, self.args.model, epoch+1)
        #      utils.save_checkpoint({
        #          'epoch': epoch + 1,
        #          'state_dict': self.model.module.state_dict(),
        #          'optimizer': self.optimizer.state_dict(),
        #          'best_pred': self.best_pred,
        #          }, self.args, is_best, filename)


    def validation(self, epoch=None):
        # Fast test during the training
        def eval_batch(model, image, target):
            r,g,b = image[:, 0, :, :]+1, image[:, 1, :, :]+1, image[:, 2, :, :]+1
            gray = 1. - (0.299*r+0.587*g+0.114*b)/2. # h, w
            gray = gray.unsqueeze(1)
            with torch.no_grad(): fake_B, _, _ = gan.netG_A.forward(image, gray)
            outputs = self.model(fake_B.clamp(-1, 1))

            # Gathers tensors from different GPUs on a specified device
            # outputs = gather(outputs, 0, dim=0)

            pred = outputs[0]
            pred = F.upsample(pred, size=(target.size(1), target.size(2)), mode='bilinear')

            correct, labeled = utils.batch_pix_accuracy(pred.data, target)
            inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)

            return correct, labeled, inter, union

        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        name2inter = {}; name2union = {}
        tbar = tqdm(self.valloader, desc='\r')

        for i, (image, target, name, class_freq) in enumerate(tbar):
            if torch_ver == "0.3":
                image = Variable(image, volatile=True)
                correct, labeled, inter, union = eval_batch(self.model, image, target)
            else:
                with torch.no_grad():
                    correct, labeled, inter, union = eval_batch(self.model, image, target)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            name2inter[name[0]] = inter.tolist()
            name2union[name[0]] = union.tolist()

            tbar.set_description('pixAcc: %.2f, mIoU: %.2f' % (pixAcc, mIoU))
        self.logger.info('pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))
        self.writer.add_scalars('IoU', {'validation iou': mIoU}, epoch)
        with open("name2inter", 'w') as fp:
            json.dump(name2inter, fp)
        with open("name2union", 'w') as fp:
            json.dump(name2union, fp)

        # cm = self.confusion_matrix_weather.get_scores()['cm']
        # self.logger.info(str(cm))
        # self.confusion_matrix_weather.reset()
        # cm = self.confusion_matrix_timeofday.get_scores()['cm']
        # self.logger.info(str(cm))
        # self.confusion_matrix_timeofday.reset()

        if epoch is not None:
            new_pred = (pixAcc + mIoU) / 2
            if new_pred > self.best_pred:
                is_best = True
                self.best_pred = new_pred
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, self.args, is_best)

if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    trainer.logger.info(['Starting Epoch:', str(args.start_epoch)])
    trainer.logger.info(['Total Epoches:', str(args.epochs)])

    if not args.eval:
        for epoch in range(args.start_epoch, args.epochs):
            trainer.training(epoch)
            if (not args.no_val) and (epoch % 10 == 0):
                trainer.validation(epoch=epoch)
    elif args.eval:
        trainer.validation()
