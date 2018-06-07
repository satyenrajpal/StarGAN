import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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

class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,k=4,s=2, p=1):
        super(ConvBlock,self).__init__()
        
        self.block=nn.Sequential(
            nn.Conv2d(in_channel, out_channel,kernel_size=k, stride=s, padding=p, bias=False),
            nn.InstanceNorm2d(out_channel, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))
        
    def forward(self,x):
        return self.block(x)

class TransposeConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,k=4,s=2, p=1):
        super().__init__()
        
        self.block=nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel,kernel_size=k, stride=s, padding=p, bias=False),
            nn.InstanceNorm2d(out_channel, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))
        
    def forward(self,x):
        return self.block(x)

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=[128,64], c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        #From_RGB layers -> convert 3+c_dim to conv_dim
        self.from_rgb=nn.ModuleList([nn.Conv2d(3+c_dim,256, kernel_size=3,padding=1, bias=False), #3x32x32 -> 256x32x32
                                     nn.Conv2d(3+c_dim,128, kernel_size=3,padding=1, bias=False), #3x64x64 -> 128x64x64
                                     nn.Conv2d(3+c_dim,64, kernel_size=3, padding=1,bias=False)])#3x32x32 -> 64x128x128
        
        # layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        # layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        # layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        self.down_sampling=nn.ModuleList([ConvBlock(128,256), #128x64x64 -> 256X32x32
                                         ConvBlock(64,128)]) #64x128x128 -> 128x64x64

        # curr_dim = conv_dim
        # for i in range(2):
        #     layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        #     layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
        #     layers.append(nn.ReLU(inplace=True))
        #     curr_dim = curr_dim * 2

        # Bottleneck layers.
        bottleneck_layers=[]
        for i in range(repeat_num):
            bottleneck_layers.append(ResidualBlock(dim_in=256, dim_out=256))

        self.bottleneck=nn.Sequential(*bottleneck_layers)
        
        # Up-sampling layers.
        # for i in range(2):
        #     layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
        #     layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
        #     layers.append(nn.ReLU(inplace=True))
        #     curr_dim = curr_dim // 2

        self.up_sampling=nn.ModuleList([TransposeConvBlock(256,128), # 32x32 -> 64x64 
                                        TransposeConvBlock(128,64)])# 64x64 -> 128x128

        # layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        # layers.append(nn.Tanh())
        # conv_toRGB=[nn.Conv2d(conv_dim[i],3, kernel_size=3, padding=1, bias=False)
        #  for i in range(len(conv_dim))]
        self.to_rgb=nn.ModuleList([nn.Conv2d(256,3,kernel_size=3,padding=1,bias=False), #256x32x32 -> 3x32x32
                                   nn.Conv2d(128,3,kernel_size=3,padding=1,bias=False), #128x64x64 -> 3x64x64
                                   nn.Conv2d(64,3,kernel_size=3,padding=1,bias=False)]) #64x128x128 -> 3x128x128

        # self.main = nn.Sequential(*layers)

    def forward(self, x, c,step=0,alpha=-1):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        x=self.from_rgb[step](x) #convert 3xAxA -> FMapxAxA
        for i,down in enumerate(self.down_sampling):
            if i<step:
                x=down(x)
        
        out=self.bottleneck(x)

        for i, up in enumerate(self.up_sampling):
            if i<step:
                out=up(out)
            #collect (upsample-1) for fade-in
            if step>0 and i==step-2:
                prev_layer=out

        out=self.to_rgb[step](out)

        if step>0 and 0<=alpha<1:
            skip_rgb=self.to_rgb[step-1](prev_layer)
            skip_rgb=F.upsample(skip_rgb,scale_factor=2)
            out=(1-alpha)*skip_rgb + alpha*out

        return out

        # return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, steps=2, ini_res=32):
        super(Discriminator, self).__init__()
        # layers = []
        # layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        # layers.append(nn.LeakyReLU(0.01))

        # curr_dim = conv_dim
        # for i in range(1, repeat_num):
        #     layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
        #     layers.append(nn.LeakyReLU(0.01))
        #     curr_dim = curr_dim * 2

        # kernel_size = int(image_size / np.power(2, repeat_num))

        self.from_rgb=nn.ModuleList([nn.Conv2d(3,256, kernel_size=3,padding=1),  #3x32x32 -> 256x32x32
                                     nn.Conv2d(3,128, kernel_size=3,padding=1),  #3x64x64 -> 128x64x64
                                     nn.Conv2d(3,64, kernel_size=3,padding=1)]) #3x128x128-> 64x128x128
        # print(self.from_rgb)

        self.progressive=nn.ModuleList([ConvBlock(128,256), #128x64^2->256x32^2 
                                        ConvBlock(64,128)]) #64x128^2->128x64^2

        # Downsample from 256x32x32 -> 512x16x16 -> 1024x8x8 -> 2048x4x4 -> 4096x2x2
        res=[2**(math.log2(conv_dim)+2+i) for i in range(int(math.log2(ini_res)))]

        block=[]
        for i in range(len(res)-1) :
            block.append(nn.Conv2d(int(res[i]), int(res[i+1]), kernel_size=4, stride=2, padding=1))
            block.append(nn.LeakyReLU(0.01))

        self.down_sample = nn.Sequential(*block)
        self.conv1 = nn.Conv2d(res[-1], 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(res[-1], c_dim, kernel_size=2, bias=False)
        
    def forward(self, x,step=0,alpha=-1):
        # h = self.main(x)
        h=self.from_rgb[step](x)
        if step>0:
            print("Size of h is ",h.size())
        for i in range(step, 0, -1):
            h=self.progressive[i-1](h)
            
            if i==step and 0<=alpha<1:
                skip_rgb=F.avg_pool2d(x,2)
                skip_rgb=self.from_rgb[step-1](skip_rgb)
                assert skip_rgb.size() == h.size()
                h=(1-alpha)*skip_rgb+alpha*h
                print("fade-in in Discriminator")
        
        assert h.size()[2]==32
        out=self.down_sample(h)

        out_src = self.conv1(out)
        out_cls = self.conv2(out)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))