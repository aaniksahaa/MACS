"""Shared convolutional building blocks used by ResUNet and DeepLabV3."""
import torch.nn as nn


def conv2d_norm_act(in_planes, planes, kernel_size=3, stride=1, groups=1,
                    dilation=1, padding=1, bias=False, pad_mode='replicate',
                    norm_mode='bn', act_mode='relu'):
    """2D convolution + normalization + activation"""
    layers = [nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride,
                        groups=groups, dilation=dilation, padding=padding,
                        padding_mode=pad_mode, bias=bias)]
    
    if norm_mode == 'bn':
        layers.append(nn.BatchNorm2d(planes))
    elif norm_mode == 'gn':
        layers.append(nn.GroupNorm(8, planes))
    elif norm_mode == 'in':
        layers.append(nn.InstanceNorm2d(planes))
    
    if act_mode == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif act_mode == 'leaky_relu':
        layers.append(nn.LeakyReLU(0.1, inplace=True))
    elif act_mode == 'elu':
        layers.append(nn.ELU(inplace=True))
    elif act_mode == 'gelu':
        layers.append(nn.GELU())
    elif act_mode != 'none':
        layers.append(nn.ReLU(inplace=True))
    
    return nn.Sequential(*layers)

class SELayer2d(nn.Module):
    """Squeeze-and-Excitation layer for 2D"""
    def __init__(self, channel, reduction=16, act_mode='relu'):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True) if act_mode == 'relu' else nn.ELU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BasicBlock2d(nn.Module):
    """Basic residual block for 2D"""
    def __init__(self, in_planes, planes, stride=1, dilation=1, groups=1,
                 projection=False, pad_mode='replicate', act_mode='elu', norm_mode='bn'):
        super().__init__()
        self.conv = nn.Sequential(
            conv2d_norm_act(in_planes, planes, kernel_size=3, dilation=dilation,
                            stride=stride, groups=groups, padding=dilation,
                            pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv2d_norm_act(planes, planes, kernel_size=3, dilation=dilation,
                            stride=1, groups=groups, padding=dilation,
                            pad_mode=pad_mode, norm_mode=norm_mode, act_mode='none')
        )

        self.projector = nn.Identity()
        if in_planes != planes or stride != 1 or projection:
            self.projector = conv2d_norm_act(
                in_planes, planes, kernel_size=1, padding=0,
                stride=stride, norm_mode=norm_mode, act_mode='none')
        
        if act_mode == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_mode == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv(x)
        y = y + self.projector(x)
        return self.act(y)
