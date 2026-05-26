import math
import time
from collections import OrderedDict
from functools import partial
from typing import Optional, Union
from models.new_model.encoders.CNN.resnet import resnet18,resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.new_model.encoders.Transformer.BiFormer.bra_legacy import BiLevelRoutingAttention
from models.new_model.encoders.Transformer.BiFormer.modules import Attention, AttentionLePE, DWConv
from models.new_model.modules import MAFI as FCM
from thop import clever_format, profile
from models.new_model.modules import LMF
def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    # if name == 'sum':
    #     return Summer(PositionalEncodingPermute2D(emb_dim))
    # elif name == 'npe.sin':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='sin')
    # elif name == 'npe.coord':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='coord')
    # elif name == 'hpe.conv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='conv', res_shortcut=True)
    # elif name == 'hpe.dsconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='dsconv', res_shortcut=True)
    # elif name == 'hpe.pointconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='pointconv', res_shortcut=True)
    else:
        raise ValueError(f'PE name {name} is not surpported!')


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                 num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 mlp_ratio=4, mlp_dwconv=False,
                 side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                                qk_scale=qk_scale, kv_per_win=kv_per_win,
                                                kv_downsample_ratio=kv_downsample_ratio,
                                                kv_downsample_kernel=kv_downsample_kernel,
                                                kv_downsample_mode=kv_downsample_mode,
                                                topk=topk, param_attention=param_attention, param_routing=param_routing,
                                                diff_routing=diff_routing, soft_routing=soft_routing,
                                                side_dwconv=side_dwconv,
                                                auto_pad=auto_pad)
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w c -> n c h w'),  # compatiability
                                      nn.Conv2d(dim, dim, 1),  # pseudo qkv linear
                                      nn.Conv2d(dim, dim, 5, padding=2, groups=dim),  # pseudo attention
                                      nn.Conv2d(dim, dim, 1),  # pseudo out linear
                                      Rearrange('n c h w -> n h w c')
                                      )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
                                 DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio * dim), dim)
                                 )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))  # (N, H, W, C)
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, H, W, C)
        else:  # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x)))  # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x)))  # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )
class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
class EDGAuX(nn.Module):
    def __init__(
        self,
        encoder_channels=(64, 64, 128),
        num_classes=8,
        out_channels=None,
        compress_c=16,
    ):
        super().__init__()
        c1, c2, c3 = encoder_channels[:3]
        base_c = c1
        out_channels = out_channels or c3
        self.conv = ConvBNReLU(in_channels=c1,out_channels=base_c)
        self.conv1 = ConvBNReLU(in_channels=c2,out_channels=base_c)
        self.conv2 = ConvBNReLU(in_channels=c3, out_channels=base_c)
        self.ema = EMA(out_channels)
        self.convfuse = Conv(out_channels,base_c,kernel_size=1)
        self.drop = nn.Dropout(0.1)
        self.convfuse1 = Conv(base_c,num_classes,kernel_size=3)
        self.edgconv_out = Conv(num_classes,1,kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.deep_score = self._add_conv(base_c,compress_c,ksize=1,stride=1)
        self.mid_score = self._add_conv(base_c,compress_c,ksize=1,stride=1)
        self.shallow_score = self._add_conv(base_c,compress_c,ksize=1,stride=1)
        self.weight_pred = nn.Conv2d(compress_c * 3,3,kernel_size=1,stride=1,padding=0)
        self.fuse_conv = self._add_conv(base_c,out_channels,ksize=3,stride=1)

    @staticmethod
    def _add_conv(in_ch, out_ch, ksize, stride, leaky=True):
        pad = (ksize - 1) // 2

        layers = [
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                bias=False
            ),
            nn.BatchNorm2d(out_ch)
        ]

        if leaky:
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        else:
            layers.append(nn.ReLU6(inplace=True))

        return nn.Sequential(*layers)

    def multi_scale_fusion(self, deep, mid, shallow):
        target_size = shallow.shape[2:]
        deep = F.interpolate(deep,size=target_size,mode="nearest")
        mid = F.interpolate(mid,size=target_size,mode="nearest")
        deep_score = self.deep_score(deep)
        mid_score = self.mid_score(mid)
        shallow_score = self.shallow_score(shallow)
        score = torch.cat(
            [deep_score, mid_score, shallow_score],
            dim=1
        )
        weight = self.weight_pred(score)
        weight = F.softmax(weight, dim=1)
        fused = (deep * weight[:, 0:1, :, :]+ mid * weight[:, 1:2, :, :]+ shallow * weight[:, 2:3, :, :])
        out = self.fuse_conv(fused)
        return out
    def forward(self, x):
        x1, x2, x3 = x
        h, w = x1.shape[2], x1.shape[3]
        e2 = self.conv(x1)
        e3 = self.conv1(x2)
        e4 = self.conv2(x3)
        feat = self.multi_scale_fusion(e4, e3, e2)
        feat = self.ema(feat)
        feat = self.convfuse(feat)
        feat = F.interpolate(
            feat,
            size=(h * 2, w * 2),
            mode="bilinear",
            align_corners=False
        )
        feat = self.drop(feat)
        feat = self.convfuse1(feat)
        feat = self.sigmoid(feat)
        few = self.edgconv_out(feat)
        few = self.sigmoid(few)
        return feat, few
class DualBiFormer(nn.Module):
    def __init__(self, depth=[3, 4, 8, 3], in_chans=3, embed_dim=[64, 128, 320, 512],
                 num_class=8,
                 pretrained=None,
                 head_dim=64, qk_scale=None,
                 drop_path_rate=0., drop_rate=0.,
                 use_checkpoint_stages=[],
                 ########
                 n_win=7,
                 kv_downsample_mode='ada_avgpool',
                 kv_per_wins=[2, 2, -1, -1],
                 topks=[8, 8, -1, -1],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[None, None, None, None],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=False,
                 # -----------------------
                 kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1],  # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios=[4, 4, 4, 4],
                 sr_ratios=[8, 4, 2, 1],
                 norm_fuse=nn.BatchNorm2d,
                 param_attention='qkvo',
                 mlp_dwconv=False):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.downsample_layers = nn.ModuleList()
        self.aux_downsample_layers = nn.ModuleList()
        # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        )
        aux_stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        )
        if (pe is not None) and 0 in pe_stages:
            stem.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
            aux_stem.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
        if use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
            aux_stem = checkpoint_wrapper(aux_stem)
        self.downsample_layers.append(stem)
        self.aux_downsample_layers.append(aux_stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i + 1])
            )
            aux_downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i + 1])
            )
            if (pe is not None) and i + 1 in pe_stages:
                downsample_layer.append(get_pe_layer(emb_dim=embed_dim[i + 1], name=pe))
                aux_downsample_layer.append(get_pe_layer(emb_dim=embed_dim[i + 1], name=pe))
            if use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
                aux_downsample_layer = checkpoint_wrapper(aux_downsample_layer)
            self.downsample_layers.append(downsample_layer)
            self.aux_downsample_layers.append(aux_downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        self.aux_stages = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.aux_norm = nn.ModuleList()

        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        topk=topks[i],
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_dim=qk_dims[i],
                        qk_scale=qk_scale,
                        kv_per_win=kv_per_wins[i],
                        kv_downsample_ratio=kv_downsample_ratios[i],
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,
                        param_attention=param_attention,
                        param_routing=param_routing,
                        diff_routing=diff_routing,
                        soft_routing=soft_routing,
                        mlp_ratio=mlp_ratios[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad) for j in range(depth[i])],
            )
            self.norm.append(nn.LayerNorm(embed_dim[i]))
            aux_stage = nn.Sequential(
                *[Block(dim=embed_dim[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        topk=topks[i],
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_dim=qk_dims[i],
                        qk_scale=qk_scale,
                        kv_per_win=kv_per_wins[i],
                        kv_downsample_ratio=kv_downsample_ratios[i],
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,
                        param_attention=param_attention,
                        param_routing=param_routing,
                        diff_routing=diff_routing,
                        soft_routing=soft_routing,
                        mlp_ratio=mlp_ratios[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad) for j in range(depth[i])],
            )
            self.aux_norm.append(nn.LayerNorm(embed_dim[i]))
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
                aux_stage = checkpoint_wrapper(aux_stage)
            self.stages.append(stage)
            self.aux_stages.append(aux_stage)
            cur += depth[i]
        self.getedge = EDGAuX(encoder_channels=(64, 64, 128), num_classes=num_class)
        self.resnet_features_main = resnet18(in_channel= in_chans, pretrained=pretrained, dilated=False)
        self.layer1 = nn.Sequential(self.resnet_features_main.conv1, self.resnet_features_main.bn1,
                                      self.resnet_features_main.relu)
        self.layer2 = nn.Sequential(self.resnet_features_main.maxpool, self.resnet_features_main.layer1)
        self.layer3 = self.resnet_features_main.layer2
        self.layer4 = self.resnet_features_main.layer3
        self.layer5 = self.resnet_features_main.layer4

        self.FCMs = nn.ModuleList([
            FCM(embed_dim[0]),
            FCM(embed_dim[1]),
            FCM(embed_dim[2]),
            FCM(embed_dim[3])])
        self.FFMs = nn.ModuleList([
             LMF(embed_dim[0]*2,embed_dim[0]),
             LMF(embed_dim[1]*2, embed_dim[1]),
             LMF(embed_dim[2]*2, embed_dim[2]),
             LMF(embed_dim[3]*2,embed_dim[3])])
        self.apply(self._init_weights)
    def _init_weights(self, m):
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
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')
    def forward_features(self, x1, x2):
        outs = []
        for i in range(4):
            x1 = self.downsample_layers[i](x1)  # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
            x1 = self.stages[i](x1)
            x2 = self.aux_downsample_layers[i](x2)  # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
            x2 = self.aux_stages[i](x2)
            x1, x2 = self.FCMs[i](x1, x2)
            x1_1 = self.norm[i](x1.permute(0, 2, 3, 1).contiguous())
            x2_1 = self.aux_norm[i](x2.permute(0, 2, 3, 1).contiguous())
            fuse = self.FFMs[i](x1_1.permute(0, 3, 1, 2).contiguous(), x2_1.permute(0, 3, 1, 2).contiguous())
            x1 = x1_1.permute(0, 3, 1, 2).contiguous()
            x2 = x2_1.permute(0, 3, 1, 2).contiguous()
            outs.append(fuse)
        return tuple(outs)

    def forward(self, x1,x2):
        x= self.forward_features(x1,x2)
        edge = []
        f1_0 = self.layer1(x1)
        f1_1 = self.layer2(f1_0)
        f1_2 = self.layer3(f1_1)
        edge.append(f1_0)
        edge.append(f1_1)
        edge.append(f1_2)
        edge= self.getedge(edge)
        return x ,edge


def load_dualpath_model(model, model_file, is_restore=False):
    # load raw state_dict
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'),weights_only=False)
        # raw_state_dict = torch.load(model_file)
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('downsample_layers') >= 0:
            state_dict[k] = v
            state_dict[k.replace('downsample_layers', 'aux_downsample_layers')] = v
        elif k.find('stages') >= 0:
            state_dict[k] = v
            state_dict[k.replace('stages', 'aux_stages')] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v
            state_dict[k.replace('norm', 'aux_norm')] = v

    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)

    del state_dict
    t_end = time.time()
    print("Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
        t_ioend - t_start, t_end - t_ioend))


class biformer_t(DualBiFormer):
    # "biformer_tiny_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHEOoGkgwgQzEDlM/root/content"
    def __init__(self, **kwargs):
        super(biformer_t, self).__init__(depth=[2, 2, 8, 2],
                                         embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
                                         n_win=8,
                                         kv_downsample_mode='identity',
                                         kv_per_wins=[-1, -1, -1, -1],
                                         topks=[1, 4, 16, -2],
                                         side_dwconv=5,
                                         before_attn_dwconv=3,
                                         layer_scale_init_value=-1,
                                         qk_dims=[64, 128, 256, 512],
                                         head_dim=32,
                                         param_routing=False, diff_routing=False, soft_routing=False,
                                         pre_norm=True,
                                         pe=None, **kwargs)


class biformer_s(DualBiFormer):
    # "biformer_small_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHDyM-x9KWRBZ832/root/content"
    def __init__(self, **kwargs):
        super(biformer_s, self).__init__(depth=[4, 4, 18, 4],
                                         embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
                                         # ------------------------------
                                         n_win=8,
                                         kv_downsample_mode='identity',
                                         kv_per_wins=[-1, -1, -1, -1],
                                         topks=[1, 4, 16, -2],
                                         side_dwconv=5,
                                         before_attn_dwconv=3,
                                         layer_scale_init_value=-1,
                                         qk_dims=[64, 128, 256, 512],
                                         head_dim=32,
                                         param_routing=False, diff_routing=False, soft_routing=False,
                                         pre_norm=True,
                                         pe=None, **kwargs)


class biformer_b(DualBiFormer):
    # "biformer_base_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHI_XPhoadjaNxtO/root/content"
    def __init__(self, **kwargs):
        super(biformer_b, self).__init__(depth=[4, 4, 18, 4],
                                         embed_dim=[96, 192, 384, 768], mlp_ratios=[3, 3, 3, 3],
                                         # use_checkpoint_stages=[0, 1, 2, 3],
                                         use_checkpoint_stages=[],
                                         # ------------------------------
                                         n_win=8,
                                         kv_downsample_mode='identity',
                                         kv_per_wins=[-1, -1, -1, -1],
                                         topks=[1, 4, 16, -2],
                                         side_dwconv=5,
                                         before_attn_dwconv=3,
                                         layer_scale_init_value=-1,
                                         qk_dims=[96, 192, 384, 768],
                                         head_dim=32,
                                         param_routing=False, diff_routing=False, soft_routing=False,
                                         pre_norm=True,
                                         pe=None, **kwargs)


