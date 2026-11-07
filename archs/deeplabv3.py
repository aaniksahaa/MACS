"""DeepLabV3-style model: simple residual encoder, ASPP head and a low-level-feature decoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import BasicBlock2d, conv2d_norm_act


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, norm_mode='bn', act_mode='elu'):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels) if norm_mode == 'bn' else nn.GroupNorm(8, out_channels),
            nn.ELU(inplace=True) if act_mode == 'elu' else nn.ReLU(inplace=True)
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_mode='bn', act_mode='elu'):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels) if norm_mode == 'bn' else nn.GroupNorm(8, out_channels),
            nn.ELU(inplace=True) if act_mode == 'elu' else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=[6, 12, 18], out_channels=256, norm_mode='bn', act_mode='elu'):
        super().__init__()
        modules = []
        modules.append(ASPPConv(in_channels, out_channels, 1, norm_mode, act_mode))
        
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate, norm_mode, act_mode))
        
        modules.append(ASPPPooling(in_channels, out_channels, norm_mode, act_mode))
        
        self.convs = nn.ModuleList(modules)
        
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels) if norm_mode == 'bn' else nn.GroupNorm(8, out_channels),
            nn.ELU(inplace=True) if act_mode == 'elu' else nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3(nn.Module):
    """DeepLabV3 style model for 2D segmentation"""
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512], 
                 atrous_rates=[6, 12, 18], norm_mode='bn', act_mode='elu'):
        super().__init__()
        
        # Encoder backbone (simple)
        self.encoder = nn.ModuleList()
        prev_ch = in_ch
        for i, feat in enumerate(features):
            if i == 0:
                self.encoder.append(conv2d_norm_act(prev_ch, feat, norm_mode=norm_mode, act_mode=act_mode))
            else:
                self.encoder.append(nn.Sequential(
                    nn.MaxPool2d(2),
                    conv2d_norm_act(prev_ch, feat, norm_mode=norm_mode, act_mode=act_mode),
                    BasicBlock2d(feat, feat, norm_mode=norm_mode, act_mode=act_mode)
                ))
            prev_ch = feat
        
        # ASPP head
        self.aspp = ASPP(features[-1], atrous_rates, 256, norm_mode, act_mode)
        
        # Low-level feature processing
        self.low_level_conv = conv2d_norm_act(features[0], 48, kernel_size=1, padding=0,
                                              norm_mode=norm_mode, act_mode=act_mode)
        
        # Decoder
        self.decoder = nn.Sequential(
            conv2d_norm_act(256 + 48, 256, norm_mode=norm_mode, act_mode=act_mode),
            conv2d_norm_act(256, 256, norm_mode=norm_mode, act_mode=act_mode),
            nn.Conv2d(256, out_ch, 1)
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        
        # Encoder
        features = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            features.append(x)
        
        # ASPP on high-level features
        x = self.aspp(features[-1])
        x = F.interpolate(x, size=features[0].shape[-2:], mode='bilinear', align_corners=True)
        
        # Low-level features
        low_level_feat = self.low_level_conv(features[0])
        
        # Combine and decode
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x
