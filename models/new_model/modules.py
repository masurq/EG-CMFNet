from timm.models.layers import trunc_normal_
import math
from typing import Optional, Sequence
import torch
import torch.nn as nn
from thop import profile, clever_format

import torch.nn.functional as F
class MAFI(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(MAFI, self).__init__()

        self.CCA1 = CrossChannelAttention(in_channels, ratio=ratio)
        self.CCA2 = CrossChannelAttention(in_channels, ratio=ratio)
        self.CSA1 = CrossSpatialAttention(kernel_size=kernel_size)
        self.CSA2 = CrossSpatialAttention(kernel_size=kernel_size)
        self.gamma1 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.gamma3 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.gamma4 = nn.Parameter(torch.zeros(1), requires_grad=True)
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x1,x2):

        x1 = self.gamma1 *(self.CCA1(x2) * x1) + x1
        x2 = self.gamma2 *(self.CCA2(x1) * x2) + x2
        x1 = self.gamma3 *(self.CSA1(x2) * x1) + x1
        x2 = self.gamma4 *(self.CSA2(x1) * x2) + x2

        return x1,x2
class CrossChannelAttention(nn.Module):

    def __init__(self, in_channels, ratio=16):
        super(CrossChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
            avg_out = self.fc(self.avg_pool(x))
            max_out = self.fc(self.max_pool(x))
            std_out = torch.std(x, dim=(2, 3), keepdim=True)
            std_out = self.fc(std_out)
            out = avg_out + max_out + std_out
            out = self.sigmoid(out)
            return out
class CrossSpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(CrossSpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        std_out = torch.std(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out, std_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out

def autopad(kernel_size: int, padding: Optional[int] = None, dilation: int = 1) -> int:
    if padding is None:
        padding = (kernel_size - 1) * dilation // 2
    return padding

def make_divisible(value: int, divisor: int = 8) -> int:
    return int((value + divisor // 2) // divisor * divisor)
class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=(norm_cfg is None),
            )
        ]

        if norm_cfg is not None:
            layers.append(self._get_norm_layer(out_channels, norm_cfg))

        if act_cfg is not None:
            layers.append(self._get_act_layer(act_cfg))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

    @staticmethod
    def _get_norm_layer(num_features, norm_cfg):
        if norm_cfg["type"] == "BN":
            return nn.BatchNorm2d(
                num_features,
                momentum=norm_cfg.get("momentum", 0.1),
                eps=norm_cfg.get("eps", 1e-5),
            )
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    @staticmethod
    def _get_act_layer(act_cfg):
        if act_cfg["type"] == "ReLU":
            return nn.ReLU(inplace=True)
        if act_cfg["type"] == "SiLU":
            return nn.SiLU(inplace=True)
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")
class DWConvgroup(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_sizes: Sequence[int] = (5, 7, 9, 11),
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(out_channels, 8)
        norm_cfg = dict(type="BN", momentum=0.03, eps=0.001)
        act_cfg = dict(type="SiLU")
        dilations = (1, 1, 1, 1)
        self.pre_conv = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.dw_convs = nn.ModuleList([
            ConvModule(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_sizes[i],
                stride=1,
                padding=autopad(kernel_sizes[i], dilation=dilations[i]),
                dilation=dilations[i],
                groups=hidden_channels,
            )
            for i in range(len(kernel_sizes))
        ])
        self.pw_conv = ConvModule(
            hidden_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
    def forward(self, x):
        x = self.pre_conv(x)
        out = x
        for conv in self.dw_convs:
            out = out + conv(x)
        out = self.pw_conv(out)
        return out

class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda
    def __repr__(self):
        return f"{self.__class__.__name__}(lambda={self.e_lambda})"
    @staticmethod
    def get_module_name():
        return "simam"
    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (
            4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
        ) + 0.5
        return x * self.activation(y)
class LMF(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        output_1 = input// 2
        self.pre_siam = SimAM()
        self.lat_siam = SimAM()
        self.fuse_siam = SimAM()
        self.dwg =DWConvgroup(
            in_channels=input,
            out_channels=output_1 * 4,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(output_1  * 4, output, kernel_size=1, padding=0),
            nn.BatchNorm2d(output_1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.Conv2d(output_1, output, kernel_size=3, padding=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
        )
    def forward(self, input1, input2):
        x = torch.cat([input1, input2], dim=1)
        cat = self.dwg(x)
        fuse = self.fuse(cat)
        input1_siam = self.pre_siam(input1)
        input2_siam = self.lat_siam(input2)
        input1_mul = input1_siam * fuse
        input2_mul = input2_siam * fuse
        fuse = self.fuse_siam(fuse)
        out = self.out(fuse + input1 + input2 + input1_mul + input2_mul)
        out = self.fuse_siam(out)
        return out



