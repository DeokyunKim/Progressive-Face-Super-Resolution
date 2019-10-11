import torch
import torch.nn as nn
from torch.nn import functional as F
from math import sqrt

"""Original EqualConv2d code is at 
   https://github.com/rosinality/style-based-gan-pytorch/blob/master/model.py
"""
    
class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name+'_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * sqrt(2/fan_in)
    
    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)

def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)
    return module

class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(EqualConv2d, self).__init__()
        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class ResBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, padding=1, stride=1):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
                            EqualConv2d(dim, dim, kernel_size=3, padding=1, stride=1),
                            nn.BatchNorm2d(dim),
                            nn.ReLU(),
                            EqualConv2d(dim, dim, kernel_size=3, padding=1, stride=1),
                            nn.BatchNorm2d(dim),
                            nn.ReLU()
                            )

    def forward(self, x):
        return self.conv(x) + x


class ConvBlock(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
                            EqualConv2d(in_plane, out_plane, kernel_size=3, padding=1, stride=1),
                            nn.LeakyReLU(0.2),
                            EqualConv2d(out_plane, out_plane, kernel_size=3, padding=1, stride=1),
                            nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):
    def __init__(self, ):
        super(Generator, self).__init__()
        step1 = [nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU()]

        step1 += [ResBlock(dim=512, kernel_size=3, stride=1, padding=1),
                  nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                  nn.BatchNorm2d(256),
                  nn.ReLU()
                  ]
        step2 = [ResBlock(dim=256, kernel_size=3, stride=1, padding=1),
                 nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                 nn.BatchNorm2d(128),
                 nn.ReLU()
                 ]

        step3 = [ResBlock(dim=128, kernel_size=3, stride=1, padding=1),
                 nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                 nn.BatchNorm2d(64),
                 nn.ReLU()]

        self.to_rgb = nn.ModuleList([nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0),
                                     nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0),
                                     nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)])

        self.step1 = nn.Sequential(*step1)
        self.step2 = nn.Sequential(*step2)
        self.step3 = nn.Sequential(*step3)
        
        #self.model = nn.Sequential(self.step1, self.step2, self.step3)

    def forward(self, input, step=1, alpha=-1):
        """Progressive generator forward"""
        if step == 1:
            out = self.step1(input)
            out = self.to_rgb[step-1](out)

        elif step == 2:
            if 0 <= alpha < 1:
                prev = self.step1(input)
                skip_rgb = F.interpolate(self.to_rgb[step-2](prev), scale_factor=2, mode='nearest')
                out = self.step2(prev)
                out = (1-alpha)*skip_rgb + alpha*self.to_rgb[step-1](out)
            else:
                out = self.step2(self.step1(input))
                out = self.to_rgb[step-1](out)

        else:
            if 0 <= alpha < 1:
                prev = self.step2(self.step1(input))
                skip_rgb = F.interpolate(self.to_rgb[step-2](prev), scale_factor=2, mode='nearest')
                out = self.step3(prev)
                out = (1-alpha)*skip_rgb + alpha*self.to_rgb[step-1](out)
            else:
                out = self.step3(self.step2(self.step1(input)))
                out = self.to_rgb[step-1](out)

        return out


class Discriminator(nn.Module):
    """Discriminator"""
    def __init__(self,):
        super(Discriminator, self).__init__()
        self.from_rgb = nn.ModuleList([
                            nn.Conv2d(3, 256, kernel_size=1, stride=1, padding=0),
                            nn.Conv2d(3, 128, kernel_size=1, stride=1, padding=0),
                            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)])

        step1 = [ConvBlock(256, 512, kernel_size=3, padding=1, stride=1), nn.AvgPool2d(kernel_size=2, stride=2)]
        step2 = [ConvBlock(128, 256, kernel_size=3, padding=1, stride=1), nn.AvgPool2d(kernel_size=2, stride=2)]
        step3 = [ConvBlock(64, 128, kernel_size=3, padding=1, stride=1), nn.AvgPool2d(kernel_size=2, stride=2)]

        self.step1 = nn.Sequential(*step1)
        self.step2 = nn.Sequential(*step2)
        self.step3 = nn.Sequential(*step3)

        #for last layer
        self.equal_conv = EqualConv2d(513, 512, kernel_size=3, stride=1, padding=1)

        self.linear = nn.Linear(512, 2048)
        self.linear2 = nn.Linear(2048, 1)

    def forward(self, input, step=1, alpha=-1):
        """Progressive discriminator forward

        Each step's output(generator output) is mixed with previous genertor output
        stacked from step1 to step3.

        | step1 -----> step2 ------> step3 |


        """
        if step == 1:#32x32
            out = self.from_rgb[step-1](input)
            out = self.step1(out)
        
        if step ==2:#64x64
            out = self.from_rgb[step-1](input)#128x64x64
            out = self.step2(out) #256x32x32
            if 0 <= alpha < 1:
                skip_rgb = F.avg_pool2d(input, kernel_size=2,stride=2)#F.interpolate(input, size=(32, 32), mode='nearest') #3x32x32
                skip_rgb = self.from_rgb[step-2](skip_rgb) #256x32x32
                out = (1-alpha)*skip_rgb + alpha * out
            out = self.step1(out) #256x16x16

        elif step ==3:#128x128
            out = self.from_rgb[step-1](input) #64x128x128
            out = self.step3(out) #128x64x64
            if 0 <= alpha < 1:
                skip_rgb = F.avg_pool2d(input, kernel_size=2,stride=2) #F.interpolate(input, size=(64, 64), mode='nearest') #3x64x64
                skip_rgb = self.from_rgb[step-2](skip_rgb) #128x64x64
                out = (1-alpha)*skip_rgb + alpha * out #128x64x64
            out = self.step2(out) #256x32x32
            out = self.step1(out) #512x16x16

        mean_std = input.std(0).mean()
        mean_std = mean_std.expand(input.size(0), 1, 16, 16)
        out = torch.cat([out, mean_std], dim=1)
        out = self.equal_conv(out)

        out = F.avg_pool2d(out, 16, stride=1)
        out = out.view(input.size(0), -1)

        out = self.linear(out)
        out = self.linear2(out)

        out = out.squeeze_(dim=1)

        return out




