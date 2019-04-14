import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
import EnlightenGAN.util.util as util
from collections import OrderedDict
from torch.autograd import Variable
import itertools
from EnlightenGAN.util.image_pool import ImagePool
from .base_model import BaseModel
import random
from . import networks
from . import fcn
import sys
import bdd.encoding.utils as utils_seg
from utils.perceptual_loss import MSSSIM

msssim_loss = MSSSIM()


def one_hot(index, classes):
    size = index.size()[:1] + (classes,) + index.size()[1:]
    view = index.size()[:1] + (1,) + index.size()[1:]

    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view)
    ones = 1.

    return mask.scatter_(1, index, ones)#[:, 1:, :, :]


class SingleModel(BaseModel):
    def name(self):
        return 'SingleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.multi_D = False
        self.mIoU_delta_mean = 0

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A_origin = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A_gray = self.Tensor(nb, 1, size, size)
        self.mask = torch.cuda.LongTensor(nb, size, size)
        # self.edges_A = torch.cuda.FloatTensor(nb, 1, size, size)
        # self.A_gt = self.Tensor(nb, opt.input_nc, size, size)
        # self.A_boundary = self.Tensor(nb, 1, size, size)

        if opt.vgg > 0:
            self.vgg_loss = networks.PerceptualLoss(opt)
            if self.opt.IN_vgg:
                self.vgg_patch_loss = networks.PerceptualLoss(opt)
                self.vgg_patch_loss.cuda()
            self.vgg_loss.cuda()
            self.vgg = networks.load_vgg16("./model", self.gpu_ids)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif opt.fcn > 0:
            self.fcn_loss = networks.SemanticLoss(opt)
            self.fcn_loss.cuda()
            self.fcn = networks.load_fcn("./model")
            self.fcn.eval()
            for param in self.fcn.parameters():
                param.requires_grad = False
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        skip = True if opt.skip > 0 else False
        # self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=skip, opt=opt)
        # if model is triain by Segmentation: DataParallel first, then DataParallel
        # if model is trained by EnlightenGAN: filter & load first, then filter & load
        self.netG_A = fcn.FCN(19, 'resnet50', aux=False, se_loss=False, dilated=True, multi_grid=True, multi_dilation=[2, 4, 8])
        self.netG_A = self.netG_A.cuda(device=self.gpu_ids[0])
        # if fix seg part: load pretrained backbone and seg head
        partial = torch.load("/home/chenwy/DynamicDeHaze/cityscapes/fcn_model/res50_di_mg_500px/model_best.pth.tar")['state_dict']
        state = self.netG_A.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state}
        print("load pretrained segmentation model of length:", len(pretrained_dict))
        state.update(pretrained_dict)
        self.netG_A = torch.nn.DataParallel(self.netG_A)
        # self.netG_A.load_state_dict(state)
        # self.netG_A.module.pretrained.eval()
        # self.netG_A.module.head.eval()

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if not self.multi_D:
                self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, False)
            else:
                self.netD_As = [networks.define_D(opt.output_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, False)
                                                for i in range(5)] # 0: road; 2: building; 8: vegetation; 10:sky; 13: car
            if self.opt.patchD:
                self.netD_P = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_patchD, opt.norm, use_sigmoid, self.gpu_ids, True)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            # self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                if self.opt.patchD:
                    self.load_network(self.netD_P, 'D_P', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            if opt.use_wgan:
                self.criterionGAN = networks.DiscLossWGANGP()
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            if opt.use_mse:
                self.criterionCycle = torch.nn.MSELoss()
            else:
                self.criterionCycle = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # fine tune seg & backbone, train header_generator
            # self.optimizer_G = torch.optim.Adam(params=[
            #     {"params": self.netG_A.module.pretrained.parameters(), "lr": opt.lr*0.1, "betas": (opt.beta1, 0.999)},
            #     {"params": self.netG_A.module.head.parameters(), "lr": opt.lr*0.1, "betas": (opt.beta1, 0.999)},
            #     {"params": self.netG_A.module.head_generator.parameters(), "lr": opt.lr, "betas": (opt.beta1, 0.999)},
            # ])
            # fixed seg & backbone, train only header_generator
            # self.optimizer_G = torch.optim.Adam(self.netG_A.module.head_generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if not self.multi_D:
                self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_D_A = torch.optim.Adam(
                    list(self.netD_As[0].parameters()) + list(self.netD_As[1].parameters()) + list(self.netD_As[2].parameters()) + list(self.netD_As[3].parameters()) + list(self.netD_As[4].parameters()),
                    lr=opt.lr, betas=(opt.beta1, 0.999))
            if self.opt.patchD:
                self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        if self.isTrain:
            if not self.multi_D:
                networks.print_network(self.netD_A)
            else:
                networks.print_network(self.netD_As[0])
            if self.opt.patchD:
                networks.print_network(self.netD_P)
        if opt.isTrain:
            self.netG_A.train()
            # self.netG_A.module.head_generator.train()
        else:
            self.netG_A.eval()
        print('-----------------------------------------------')


    def set_input(self, input, domainB=True):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_img = input['input_img']
        input_A_gray = input['A_gray']

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A_gray.resize_(input_A_gray.size()).copy_(input_A_gray)
        if domainB:
            input_B = input['B' if AtoB else 'A']
            self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.mask.resize_(input['mask'].size()).copy_(input['mask'])
        # self.edges_A.resize_(input['edges_A'].size()).copy_(input['edges_A'])
        # self.A_gt.resize_(input['A_gt'].size()).copy_(input['A_gt'])
        # self.A_boundary.resize_(input['A_boundary'].size()).copy_(input['A_boundary'])
        self.category = input['category']


    def set_input_A(self, A, A_gray):#, edges_A):
        self.input_A.resize_(A.size()).copy_(A)
        self.input_A_gray.resize_(A_gray.size()).copy_(A_gray)
        # self.edges_A.resize_(edges_A.size()).copy_(edges_A)


    def set_input_A_origin(self, A):
        self.input_A_origin.resize_(A.size()).copy_(A)

    
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_gray = Variable(self.input_A_gray, volatile=True)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        # print(np.transpose(self.real_A.data[0].cpu().float().numpy(),(1,2,0))[:2][:2][:])
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray)
        else:
            self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray)
        # self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)


    def predict(self, seg=None):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_gray = Variable(self.input_A_gray, volatile=True)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A, self.seg_real_A = self.netG_A.forward(self.input_A, self.input_A_gray)
            self.fake_B = self.input_A_origin + self.latent_real_A
            self.fake_B = self.fake_B.clamp(-1, 1)
            if seg: self.fake_B_Seg = F.softmax(seg(self.fake_B.clamp(-1, 1))[0], dim=1)
        else:
            self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray)

        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        A_gray = util.atten2im(self.real_A_gray.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B)]), self.fake_B

    # get image paths
    def get_image_paths(self):
        return self.image_paths


    def forward(self, seg, epoch):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_A_gray = Variable(self.input_A_gray)
        self.real_img = Variable(self.input_img)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        if self.opt.skip == 1:
            # self.A_gt_Seg = seg(self.A_gt.clamp(-1, 1))[0]
            self.real_A_Seg = (F.softmax(seg(self.real_A.clamp(-1, 1))[0], dim=1) + F.softmax(seg(self.real_A.clamp(-1, 1).flip(3))[0], dim=1).flip(3)) / 2
            self.fake_B, self.latent_real_A, self.seg_real_A = self.netG_A.forward(self.real_A, self.real_A_gray) # seg_real_A is the seg predicted from G it self (multi-tasking)
            self.fake_B_Seg = F.softmax(seg(self.fake_B.clamp(-1, 1))[0], dim=1) + F.softmax(seg(self.fake_B.clamp(-1, 1).flip(3))[0], dim=1).flip(3)
        else:
            self.fake_B = self.netG_A.forward(self.real_img, self.real_A_gray)
        if self.opt.patchD:
            h = self.real_A.size(2)
            w = self.real_A.size(3)
            self.fake_patch = torch.empty(self.fake_B.size(0), self.fake_B.size(1), self.opt.patchSize, self.opt.patchSize).cuda()
            self.real_patch = torch.empty(self.fake_B.size(0), self.fake_B.size(1), self.opt.patchSize, self.opt.patchSize).cuda()
            self.input_patch = torch.empty(self.fake_B.size(0), self.fake_B.size(1), self.opt.patchSize, self.opt.patchSize).cuda()

            for i in range(self.fake_B.size(0)):
                w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
                h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))
                unique, counts = np.unique(self.mask[i, h_offset:h_offset + self.opt.patchSize, w_offset:w_offset + self.opt.patchSize].detach().cpu().numpy().astype('int32'), return_counts=True)
                self.fake_patch[i] = self.fake_B[i,:, h_offset:h_offset + self.opt.patchSize, w_offset:w_offset + self.opt.patchSize]
                self.real_patch[i] = self.real_B[i,:, h_offset:h_offset + self.opt.patchSize, w_offset:w_offset + self.opt.patchSize]
                self.input_patch[i] = self.real_A[i,:, h_offset:h_offset + self.opt.patchSize, w_offset:w_offset + self.opt.patchSize]

        if self.opt.patchD_3 > 0:
            self.fake_patch_1 = []
            self.real_patch_1 = []
            self.input_patch_1 = []
            w = self.real_A.size(3)
            h = self.real_A.size(2)
            for _ in range(self.opt.patchD_3):
                self.fake_patch_tmp = torch.empty(self.fake_B.size(0), self.fake_B.size(1), self.opt.patchSize, self.opt.patchSize).cuda()
                self.real_patch_tmp = torch.empty(self.fake_B.size(0), self.fake_B.size(1), self.opt.patchSize, self.opt.patchSize).cuda()
                self.input_patch_tmp = torch.empty(self.fake_B.size(0), self.fake_B.size(1), self.opt.patchSize, self.opt.patchSize).cuda()
                for i in range(self.fake_B.size(0)):
                    w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
                    h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))
                    unique, counts = np.unique(self.mask[i, h_offset:h_offset + self.opt.patchSize, w_offset:w_offset + self.opt.patchSize].detach().cpu().numpy().astype('int32'), return_counts=True)
                    self.fake_patch_tmp[i] = self.fake_B[i,:, h_offset:h_offset + self.opt.patchSize, w_offset:w_offset + self.opt.patchSize]
                    self.real_patch_tmp[i] = self.real_B[i,:, h_offset:h_offset + self.opt.patchSize, w_offset:w_offset + self.opt.patchSize]
                    self.input_patch_tmp[i] = self.real_A[i,:, h_offset:h_offset + self.opt.patchSize, w_offset:w_offset + self.opt.patchSize]
                self.fake_patch_1.append(self.fake_patch_tmp)
                self.real_patch_1.append(self.real_patch_tmp)
                self.input_patch_1.append(self.input_patch_tmp)

    def backward_G(self, epoch, seg_criterion=None, A_gt=False):
        # self.loss_G_A = torch.zeros(1).cuda()
        if not self.multi_D:
            pred_fake = self.netD_A.forward(self.fake_B)
            if self.opt.use_wgan:
                self.loss_G_A = -pred_fake.mean()
            elif self.opt.use_ragan:
                pred_real = self.netD_A.forward(self.real_B)
                self.loss_G_A = (self.criterionGAN(pred_real - torch.mean(pred_fake), False) + self.criterionGAN(pred_fake - torch.mean(pred_real), True)) / 2
            else:
                self.loss_G_A = self.criterionGAN(pred_fake, True)
        else:
            self.loss_G_A = 0
            for c in range(5):
                # select by category; if empty: tensor([])
                if (self.category == c).nonzero().size(0) == 0: continue
                pred_fake = self.netD_As[c].forward(torch.index_select(self.fake_B, 0, (self.category == c).nonzero().view(-1).type(torch.cuda.LongTensor)))
                if self.opt.use_wgan:
                    self.loss_G_A += -pred_fake.mean()
                elif self.opt.use_ragan:
                    pred_real = self.netD_As[c].forward(torch.index_select(self.real_B, 0, (self.category == c).nonzero().view(-1).type(torch.cuda.LongTensor)))
                    self.loss_G_A += (self.criterionGAN(pred_real - torch.mean(pred_fake), False) + self.criterionGAN(pred_fake - torch.mean(pred_real), True)) / 2
                else:
                    self.loss_G_A += self.criterionGAN(pred_fake, True)
        
        loss_G_A = 0
        if self.opt.patchD:
            pred_fake_patch = self.netD_P.forward(self.fake_patch)
            if self.opt.hybrid_loss:
                loss_G_A += self.criterionGAN(pred_fake_patch, True)
            else:
                pred_real_patch = self.netD_P.forward(self.real_patch)
                loss_G_A += (self.criterionGAN(pred_real_patch - torch.mean(pred_fake_patch), False) + self.criterionGAN(pred_fake_patch - torch.mean(pred_real_patch), True)) / 2
        if self.opt.patchD_3 > 0:   
            for i in range(self.opt.patchD_3):
                pred_fake_patch_1 = self.netD_P.forward(self.fake_patch_1[i])
                if self.opt.hybrid_loss:
                    loss_G_A += self.criterionGAN(pred_fake_patch_1, True)
                else:
                    pred_real_patch_1 = self.netD_P.forward(self.real_patch_1[i])
                    loss_G_A += (self.criterionGAN(pred_real_patch_1 - torch.mean(pred_fake_patch_1), False) + self.criterionGAN(pred_fake_patch_1 - torch.mean(pred_real_patch_1), True)) / 2
                    
            if not self.opt.D_P_times2:
                self.loss_G_A += loss_G_A/float(self.opt.patchD_3 + 1)
            else:
                self.loss_G_A += loss_G_A/float(self.opt.patchD_3 + 1)*2
        else:
            if not self.opt.D_P_times2:
                self.loss_G_A += loss_G_A
            else:
                self.loss_G_A += loss_G_A*2
                
        self.loss_G = self.loss_G_A
        if epoch < 0:
            vgg_w = 0
        else:
            if seg_criterion is None: vgg_w = 1
            else: vgg_w = 0
        if vgg_w > 0:
            if self.opt.vgg > 0:
                self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg, self.fake_B, self.real_A) * self.opt.vgg if self.opt.vgg > 0 else 0
                if self.opt.patch_vgg:
                    if not self.opt.IN_vgg:
                        loss_vgg_patch = self.vgg_loss.compute_vgg_loss(self.vgg, self.fake_patch, self.input_patch) * self.opt.vgg
                    else:
                        loss_vgg_patch = self.vgg_patch_loss.compute_vgg_loss(self.vgg, self.fake_patch, self.input_patch) * self.opt.vgg
                    if self.opt.patchD_3 > 0:
                        for i in range(self.opt.patchD_3):
                            if not self.opt.IN_vgg:
                                loss_vgg_patch += self.vgg_loss.compute_vgg_loss(self.vgg, self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                            else:
                                loss_vgg_patch += self.vgg_patch_loss.compute_vgg_loss(self.vgg, self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                        self.loss_vgg_b += loss_vgg_patch/float(self.opt.patchD_3 + 1)
                    else:
                        self.loss_vgg_b += loss_vgg_patch
                self.loss_G = self.loss_G_A + self.loss_vgg_b*vgg_w
            elif self.opt.fcn > 0:
                self.loss_fcn_b = self.fcn_loss.compute_fcn_loss(self.fcn, self.fake_B, self.real_A) * self.opt.fcn if self.opt.fcn > 0 else 0
                if self.opt.patchD:
                    loss_fcn_patch = self.fcn_loss.compute_vgg_loss(self.fcn, self.fake_patch, self.input_patch) * self.opt.fcn
                    if self.opt.patchD_3 > 0:
                        for i in range(self.opt.patchD_3):
                            loss_fcn_patch += self.fcn_loss.compute_vgg_loss(self.fcn, self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.fcn
                        self.loss_fcn_b += loss_fcn_patch/float(self.opt.patchD_3 + 1)
                    else:
                        self.loss_fcn_b += loss_fcn_patch
                self.loss_G = self.loss_G_A + self.loss_fcn_b*vgg_w
            # self.loss_G = self.L1_AB + self.L1_BA

        ## Seg Loss ################################
        if seg_criterion is not None:
            # mIoU of enhanced image
            inter, union = utils_seg.batch_intersection_union(self.fake_B_Seg.data, self.mask, 19)
            idx = union > 0
            IoU = 1.0 * inter[idx] / (np.spacing(1) + union[idx])
            self.mIoU = np.nan_to_num(IoU.mean())

            with torch.no_grad():
                # mIoU of origin image by pretrained Seg Model
                inter, union = utils_seg.batch_intersection_union(self.real_A_Seg.data, self.mask, 19)
                idx = union > 0
                IoU = 1.0 * inter[idx] / (np.spacing(1) + union[idx])
                self.mIoU_ori = np.nan_to_num(IoU.mean())
                self.mIoU_delta_mean = 0.8 * self.mIoU_delta_mean + 0.2 * np.round(self.mIoU-self.mIoU_ori, 3) 

                # mIoU of origin image by Generator
                inter, union = utils_seg.batch_intersection_union(self.seg_real_A.data, self.mask, 19)
                idx = union > 0
                IoU = 1.0 * inter[idx] / (np.spacing(1) + union[idx])
                print("mIoU_generator", np.round(np.nan_to_num(IoU.mean()), 3))

            print("G:", self.loss_G.data[0], "mIoU gain:", np.round(self.mIoU-self.mIoU_ori, 3), "mean:", np.round(self.mIoU_delta_mean, 3), "lum:", 255*(1 - self.input_A_gray).mean(), "epoch:", epoch)

            lambd = 3
            self.loss_Seg = seg_criterion(self.fake_B_Seg, self.mask) + lambd * seg_criterion(self.seg_real_A, self.mask)
            self.loss_G += self.loss_Seg
        ############################################
        ## GAN_GT Loss ################################
        if A_gt:
            # msssim = msssim_loss((self.fake_B.clamp(-1, 1)+1)/2*255, (self.A_gt+1)/2*255, weight_map=self.A_boundary)
            l1 = (F.l1_loss((self.fake_B+1)/2*255, (self.A_gt+1)/2*255, reduction='none') * self.A_boundary).mean()
            # self.loss_gt = 3 * msssim + 0.16 * l1
            self.loss_gt = 0.1 * l1
            print("loss_gt", self.loss_gt.data[0])
            self.loss_G += self.loss_gt
        ############################################

        self.loss_G.backward(retain_graph=True)

    def backward_D_basic(self, netD, real, fake, use_ragan):
        # Real
        pred_real = netD.forward(real)
        pred_fake = netD.forward(fake.detach())
        if self.opt.use_wgan:
            loss_D_real = pred_real.mean()
            loss_D_fake = pred_fake.mean()
            loss_D = loss_D_fake - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD, real.data, fake.data)
        elif self.opt.use_ragan and use_ragan:
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D.backward()
        return loss_D

    def backward_D_A(self):
        # fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B = self.fake_B
        if not self.multi_D:
            self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, True)
        else:
            self.loss_D_A = 0
            for c in range(5):
                # select by category; if empty: tensor([])
                if (self.category == c).nonzero().size(0) == 0: continue
                self.loss_D_A += self.backward_D_basic(self.netD_As[c], torch.index_select(self.real_B, 0, (self.category == c).nonzero().view(-1).type(torch.cuda.LongTensor)), torch.index_select(fake_B, 0, (self.category == c).nonzero().view(-1).type(torch.cuda.LongTensor)), True)
        self.loss_D_A.backward()
    
    def backward_D_P(self):
        if self.opt.hybrid_loss:
            loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch, self.fake_patch, False)
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P += self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i], False)
                self.loss_D_P = loss_D_P/float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P = loss_D_P
        else:
            loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch, self.fake_patch, True)
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P += self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i], True)
                self.loss_D_P = loss_D_P/float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P = loss_D_P
        if self.opt.D_P_times2:
            self.loss_D_P = self.loss_D_P*2
        self.loss_D_P.backward()


    def optimize_parameters(self, epoch, step, seg=None, seg_criterion=None):
        late_update = 2
        # forward
        self.forward(seg, epoch)
        # G_A and G_B
        self.backward_G(epoch, seg_criterion=seg_criterion, A_gt=False)
        if step % late_update == 0: # late update
            self.optimizer_G.step()
            self.optimizer_G.zero_grad()
        # D_A
        self.backward_D_A()
        if self.opt.patchD: self.backward_D_P()
        if step % late_update == 0:
            self.optimizer_D_A.step()
            self.optimizer_D_A.zero_grad()
            if self.opt.patchD:
                self.optimizer_D_P.step()
                self.optimizer_D_P.zero_grad()


    def get_current_errors(self, epoch):
        D_A = self.loss_D_A.data[0]
        D_P = self.loss_D_P.data[0] if self.opt.patchD else 0
        G_A = self.loss_G_A.data[0]
        if self.opt.vgg > 0:
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ("D_P", D_P), ("mIoU gain", np.round(self.mIoU-self.mIoU_ori, 3))])
        elif self.opt.fcn > 0:
            fcn = self.loss_fcn_b.data[0]/self.opt.fcn if self.opt.fcn > 0 else 0
            if 'images' in self.opt.name:
                return OrderedDict([('D_A', D_A), ('G_A', G_A), ("fcn", fcn), ("D_P", D_P)])
            else:
                return OrderedDict([('D_A', D_A), ('G_A', G_A), ("fcn", fcn), ("GT", self.loss_gt.data[0]), ("D_P", D_P), ("mIoU gain", np.round(self.mIoU-self.mIoU_ori, 3))])
        

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        if self.opt.skip > 0:
            latent_real_A = util.tensor2im(self.latent_real_A.data)
            latent_show = util.latent2im(self.latent_real_A.data)
            if self.opt.patchD:
                fake_patch = util.tensor2im(self.fake_patch.data)
                real_patch = util.tensor2im(self.real_patch.data)
                if self.opt.patch_vgg:
                    input_patch = util.tensor2im(self.input_patch.data)
                    if not self.opt.self_attention:
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch), ('input_patch', input_patch)])
                    else:
                        # currently into here
                        self_attention = util.atten2im(self.real_A_gray.data)
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch), ('input_patch', input_patch), ('self_attention', self_attention)
                                ])
                else:
                    if not self.opt.self_attention:
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch)])
                    else:
                        self_attention = util.atten2im(self.real_A_gray.data)
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch), ('self_attention', self_attention)])
            else:
                if not self.opt.self_attention:
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B)])
                else:
                    self_attention = util.atten2im(self.real_A_gray.data)
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
                                    ('latent_real_A', latent_real_A), ('latent_show', latent_show),
                                    ('self_attention', self_attention)])
        else:
            if not self.opt.self_attention:
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])
            else:
                self_attention = util.atten2im(self.real_A_gray.data)
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
                                    ('self_attention', self_attention)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        if not self.multi_D:
            self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        else:
            for c in range(5):
                self.save_network(self.netD_As[c], 'D_A'+str(c), label, self.gpu_ids)
        if self.opt.patchD:
            self.save_network(self.netD_P, 'D_P', label, self.gpu_ids)

    def update_learning_rate(self):
        
        if self.opt.new_lr:
            lr = self.old_lr/2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        if self.opt.patchD:
            for param_group in self.optimizer_D_P.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
