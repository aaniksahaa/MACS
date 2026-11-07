"""2D residual U-Net (ResUNet)."""
import torch.nn as nn
import torch.nn.functional as F

from .blocks import BasicBlock2d, conv2d_norm_act


class ResUNet(nn.Module):
    """2D Residual U-Net with flexible configuration"""
    def __init__(self, in_ch=1, out_ch=1, features=[32, 64, 128, 256, 512],
                 block_type='residual', norm_mode='bn', act_mode='elu', pooling=False):
        super().__init__()
        self.depth = len(features)
        self.pooling = pooling
        
        # Input layer
        self.conv_in = conv2d_norm_act(in_ch, features[0], kernel_size=5, padding=2,
                                       norm_mode=norm_mode, act_mode=act_mode)
        
        # Encoder
        self.down_layers = nn.ModuleList()
        for i in range(self.depth):
            if i == 0:
                layer = BasicBlock2d(features[0], features[0], norm_mode=norm_mode, act_mode=act_mode)
            else:
                stride = 1 if pooling else 2
                layer = nn.Sequential(
                    nn.MaxPool2d(2) if pooling else nn.Identity(),
                    conv2d_norm_act(features[i-1], features[i], stride=stride,
                                    norm_mode=norm_mode, act_mode=act_mode),
                    BasicBlock2d(features[i], features[i], norm_mode=norm_mode, act_mode=act_mode)
                )
            self.down_layers.append(layer)
        
        # Decoder
        self.up_layers = nn.ModuleList()
        for j in range(1, self.depth):
            layer = nn.ModuleList([
                conv2d_norm_act(features[j], features[j-1], kernel_size=3, padding=1,
                                norm_mode=norm_mode, act_mode=act_mode),
                BasicBlock2d(features[j-1], features[j-1], norm_mode=norm_mode, act_mode=act_mode)
            ])
            self.up_layers.append(layer)
        
        # Output layer
        self.conv_out = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        
        down_x = [None] * (self.depth - 1)
        for i in range(self.depth - 1):
            x = self.down_layers[i](x)
            down_x[i] = x
        
        x = self.down_layers[-1](x)
        
        for j in range(self.depth - 1):
            i = self.depth - 2 - j
            x = self.up_layers[i][0](x)
            x = self._upsample_add(x, down_x[i])
            x = self.up_layers[i][1](x)
        
        return self.conv_out(x)
    
    def _upsample_add(self, x, y):
        x = F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=True)
        return x + y
