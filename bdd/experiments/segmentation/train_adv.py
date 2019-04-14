###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import os
import copy
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses, BatchNorm2d
from encoding.nn import SegmentationMultiLosses
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_segmentation_dataset
from encoding.models import get_segmentation_model


from option import Options


torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

class Trainer():
    def __init__(self, args):
        self.args = args
        args.log_name = str(args.checkname)
        self.logger = utils.create_logger(args.log_root, args.log_name)
        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),
            # transform.Normalize([.485, .456, .406], [.229, .224, .225])
            ])
        # dataset
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size, 'logger': self.logger, 'scale': args.scale}
        trainset = get_segmentation_dataset(args.dataset, split='train', mode='train', **data_kwargs)
        testset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False, **kwargs)
        self.nclass = trainset.num_class
        # model
        model = get_segmentation_model(args.model, dataset=args.dataset, backbone=args.backbone, aux=args.aux, se_loss=args.se_loss,
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
        params_list.append({'params': model.weather_classifier.parameters(), 'lr': 0 * args.lr*10})
        params_list.append({'params': model.time_classifier.parameters(), 'lr': 0 * args.lr*10})
        optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # self.criterion = SegmentationMultiLosses(nclass=self.nclass)
        self.criterion = SegmentationLosses(se_loss=args.se_loss, aux=args.aux, nclass=self.nclass)
        # self.criterion = torch.nn.CrossEntropyLoss()
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

    def training(self, epoch):
        train_loss = 0.0

        ################################################
        self.model.train()
        self.model.module.weather_classifier.eval()
        self.model.module.time_classifier.eval()
        # self.model.eval()
        # self.model.module.weather_classifier.train()
        # self.model.module.time_classifier.train()
        ################################################

        tbar = tqdm(self.trainloader)

        for i, (image, target, weather, timeofday, scene) in enumerate(tbar):
            weather = weather.cuda(); timeofday = timeofday.cuda()
            ################################################
            # self.scheduler(self.optimizer, i, epoch, self.best_pred)
            ################################################
            self.optimizer.zero_grad()
            if torch_ver == "0.3":
                image = Variable(image)
                target = Variable(target)
            outputs, weather_o, timeofday_o = self.model(image)

            # create weather / timeofday target mask #######################
            b, _, h, w = weather_o.size()
            weather_t = torch.ones((b, h, w)).long().cuda()
            for bi in range(b): weather_t[bi] *= weather[bi]
            timeofday_t = torch.ones((b, h, w)).long().cuda()
            for bi in range(b): timeofday_t[bi] *= timeofday[bi]
            ################################################################

            loss = self.criterion(outputs, target)
            # loss = self.criterion(weather_o, weather_t) + self.criterion(timeofday_o, timeofday_t)

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
        self.logger.info('Train loss: %.3f' % (train_loss / (i + 1)))

        # save checkpoint every 5 epoch
        is_best = False
        if epoch % 5 == 0:
            # filename = "checkpoint_%s.pth.tar"%(epoch+1)
            filename = "checkpoint_%s.%s.%s.%s.pth.tar"%(self.args.log_root, self.args.checkname, self.args.model, epoch+1)
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                }, self.args, is_best, filename)


    def validation(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, target, weather, timeofday, scene):
            weather = weather.cuda(); timeofday = timeofday.cuda()
            outputs, weather_o, timeofday_o = model(image)
            # Gathers tensors from different GPUs on a specified device
            # outputs = gather(outputs, 0, dim=0)
            pred = outputs[0]
            target = target.cuda()

            b, _, h, w = weather_o.size()
            weather_t = torch.ones((b, h, w)).long().cuda()
            for bi in range(b): weather_t[bi] *= weather[bi]
            timeofday_t = torch.ones((b, h, w)).long().cuda()
            for bi in range(b): timeofday_t[bi] *= timeofday[bi]

            correct, labeled = utils.batch_pix_accuracy(pred.data, target)
            inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)

            correct_weather, labeled_weather = utils.batch_pix_accuracy(weather_o.data, weather_t)
            correct_timeofday, labeled_timeofday = utils.batch_pix_accuracy(timeofday_o.data, timeofday_t)
            return correct, labeled, inter, union, correct_weather, labeled_weather, correct_timeofday, labeled_timeofday

        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        total_correct_weather = 0; total_label_weather = 0; total_correct_timeofday = 0; total_label_timeofday = 0
        tbar = tqdm(self.valloader, desc='\r')

        for i, (image, target, weather, timeofday, scene) in enumerate(tbar):
            if torch_ver == "0.3":
                image = Variable(image, volatile=True)
                correct, labeled, inter, union, correct_weather, labeled_weather, correct_timeofday, labeled_timeofday = eval_batch(self.model, image, target, weather, timeofday, scene)
            else:
                with torch.no_grad():
                    correct, labeled, inter, union, correct_weather, labeled_weather, correct_timeofday, labeled_timeofday = eval_batch(self.model, image, target, weather, timeofday, scene)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()

            total_correct_weather += correct_weather
            total_label_weather += labeled_weather
            pixAcc_weather = 1.0 * total_correct_weather / (np.spacing(1) + total_label_weather)
            total_correct_timeofday += correct_timeofday
            total_label_timeofday += labeled_timeofday
            pixAcc_timeofday = 1.0 * total_correct_timeofday / (np.spacing(1) + total_label_timeofday)

            tbar.set_description('pixAcc: %.3f, mIoU: %.3f, pixAcc_weather: %.3f, pixAcc_timeofday: %.3f' % (pixAcc, mIoU, pixAcc_weather, pixAcc_timeofday))
        self.logger.info('pixAcc: %.3f, mIoU: %.3f, pixAcc_weather: %.3f, pixAcc_timeofday: %.3f' % (pixAcc, mIoU, pixAcc_weather, pixAcc_timeofday))

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

    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        if not args.no_val:
            trainer.validation(epoch)
