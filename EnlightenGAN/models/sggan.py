import torch
import torch.nn as nn
from EnlightenGAN.lib.nn import SynchronizedBatchNorm2d as SynBN2d
import torch.nn.functional as F
import numpy as np

norm_layer = SynBN2d
# norm_layer = nn.InstanceNorm2d

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(dim_out),
            # norm_layer(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(dim_out)
            # norm_layer(dim_out, affine=True)
        )

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, s_dim=7, repeat_num=6):
        super(Generator, self).__init__()
        print('initializing generator')
        print(c_dim)
        print(s_dim)
        layers = []
        layers.append(nn.Conv2d(3+c_dim+s_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c, s):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        x = torch.cat([x,s], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        c_dim=5
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        out_aux = self.conv2(h)
        return out_real.squeeze(), out_aux.squeeze()

class Segmentor(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, repeat_num=4):
        super(Segmentor, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 7, kernel_size=7, stride=1, padding=3, bias=False))
        # layers.append(nn.LogSoftmax())
        # layers.append(nn.Softmax2d())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def define_G(gpu_ids):
    netG = Generator_Segmentor()
    netG.cuda(device=gpu_ids[0])
    netG = torch.nn.DataParallel(netG)#, gpu_ids)
    netG.apply(weights_init)
    return netG


class Generator_Segmentor(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, repeat_num=6):
        super(Generator_Segmentor, self).__init__()
        print('initializing generator')
        layers = []
        layers.append(nn.Conv2d(3+1, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(norm_layer(conv_dim))
        # layers.append(norm_layer(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(norm_layer(curr_dim*2))
            # layers.append(norm_layer(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        
        self.main = nn.Sequential(*layers)

        layers_generator = []; layers_segmentor = []
        # Up-Sampling
        for i in range(2):
            layers_generator.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers_generator.append(norm_layer(curr_dim//2))
            # layers_generator.append(norm_layer(curr_dim//2, affine=True))
            layers_generator.append(nn.ReLU(inplace=True))

            layers_segmentor.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers_segmentor.append(norm_layer(curr_dim//2))
            # layers_segmentor.append(norm_layer(curr_dim//2, affine=True))
            layers_segmentor.append(nn.ReLU(inplace=True))

            curr_dim = curr_dim // 2

        layers_generator.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_segmentor.append(nn.Conv2d(curr_dim, 19, kernel_size=7, stride=1, padding=3, bias=False))

        self.generator = nn.Sequential(*layers_generator)
        self.segmentor = nn.Sequential(*layers_segmentor)

    def forward(self, input, gray):
        feature = self.main(torch.cat([input, gray], dim=1))
        latent = gray * self.generator(feature)
        output = latent + input
        seg = self.segmentor(feature)
        return output, latent, seg
