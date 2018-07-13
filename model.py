import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.main=nn.Sequential(*layers)

        upsample=[]
        # Up-sampling layers.
        for i in range(2):
            upsample.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            upsample.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            upsample.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        upsample.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        upsample.append(nn.Tanh())
        self.upsample = nn.Sequential(*upsample)

    def forward(self, x, c=None,return_interp=False,partial=False):
        # Replicate spatially and concatenate domain information.
        if partial and c is None:
            x=self.upsample(x)
            return x
        
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        h=self.main(x)
        x=self.upsample(h)

        if return_interp:
            return x,h
        return x
        

class FE(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, repeat_num=6):
        super(FE, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = image_size // np.power(2, repeat_num)
        self.main = nn.Sequential(*layers)
        # self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        # h = self.main(x)
        # out_src = self.conv1(h)
        # out_cls = self.conv2(h)
        # return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
        return self.main(x)

class Discriminator(nn.Module):
    """ Outputs attributes and real? logits"""
    def __init__(self,image_size=128,conv_dim=64,c_dim=5, repeat_num=6):
        super(Discriminator,self).__init__()
        
        curr_dim=conv_dim
        for _ in range(1,repeat_num):
            curr_dim = curr_dim*2
        
        kernel_size=image_size//np.power(2,repeat_num)

        self.real_conv=nn.Conv2d(curr_dim,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.cls_conv=nn.Conv2d(curr_dim,c_dim,kernel_size=kernel_size,stride=1,bias=False)

    def forward(self,x):
        out_src=self.real_conv(x)
        out_cls=self.cls_conv(x)
        return out_src,out_cls.view(out_cls.size(0),out_cls.size(1))

class Q(nn.Module):
    """ Outputs logits and stats for G(x,c)"""
    def __init__(self,image_size=128,conv_dim=64,repeat_num=6,con_dim=2):
        super(Q,self).__init__()

        curr_dim=conv_dim
        for _ in range(1,repeat_num):
            curr_dim=conv_dim*2

        self.conv=nn.Sequential(nn.Conv2d(curr_dim, 128,  kernel_size=1,bias=False),
                                nn.LeakyReLU(0.01,inplace=True),
                                nn.Conv2d(128,    64, kernel_size=1,bias=False),
                                nn.LeakyReLU(0.01,inplace=True))

        self.conv_mu =nn.Conv2d(64,con_dim,kernel_size=1)
        self.conv_var=nn.Conv2d(64,con_dim,kernel_size=1)

    def forward(self,h):
        out=self.conv(h)
        mu_out=self.conv_mu(out).squeeze()
        var_out=self.conv_var(out).squeeze().exp()

        return mu_out,var_out

