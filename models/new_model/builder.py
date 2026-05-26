import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from functools import partial

from models.new_model.init_func import init_weight, nostride_dilate, patch_first_conv_mit, patch_first_conv_swin, \
    patch_first_conv_biformer, patch_first_conv_SMT, patch_first_conv_DilateFormer, patch_first_conv_resnet, \
    patch_first_conv_cswin, patch_first_conv_mobilenet, patch_first_conv_EMO, patch_first_conv_single_biformer

from utils.Common import common
from utils.Logger import Logger

from thop import clever_format, profile
from config.opt_sar_config512 import config
# from config.korea import config

class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, logger=None, norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.channels2 = [96, 192, 384, 768]
        self.norm_layer = norm_layer
        self.logger = logger
        # import backbone and decoder
        if cfg.backbone == 'biformer_t':
            logger.log('INFO', 'Using backbone: Biformer-Tiny', show_time=False)
            self.channels = [64, 128, 256, 512]
            from models.new_model.encoders.Transformer.BiFormer.dual_biformer import biformer_t as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'biformer_s':
            logger.log('INFO', 'Using backbone: Biformer-Small', show_time=False)
            self.channels = [64, 128, 256, 512]
            from models.new_model.encoders.Transformer.BiFormer.dual_biformer import biformer_s as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'biformer_b':
            logger.log('INFO', 'Using backbone: Biformer-Base', show_time=False)
            self.channels = [96, 192, 384, 768]
            from models.new_model.encoders.Transformer.BiFormer.dual_biformer import biformer_b as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        else:
            logger.log('INFO', 'Using backbone: Biformer-Tiny', show_time=False)
            self.channels = [64, 128, 256, 512]
            from models.new_model.encoders.Transformer.BiFormer.dual_biformer import biformer_t as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        self.aux_head = None

        if cfg.decoder == 'MLPAlignDecoder':
            logger.log('INFO', 'Using MLP Aligned Decoder', show_time=False)
            from decoders.MLPAlignDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.num_classes,
                                           norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)
        else:
            logger.log('INFO', 'No decoder(FCN-32s)', show_time=False)
            from decoders.fcnhead import MainFCNHead
            self.decode_head = MainFCNHead(in_channels=self.channels[-1], num_classes=cfg.num_classes,
                                           norm_layer=norm_layer)

        self.init_weights(cfg, pretrained=cfg.pretrained_model)

    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            self.logger.log('INFO', 'Loading pretrained model: {}'.format(pretrained), show_time=False)
            self.backbone.init_weights(pretrained=pretrained)

        if 'biformer' in cfg.backbone or 'convnext' in cfg.backbone:
            # patch_first_conv_single_biformer(self, cfg.in_channel1, cfg.in_channel2)
            patch_first_conv_biformer(self, cfg.in_channel1, cfg.in_channel2)
        self.logger.log('INFO', 'Initing weights ...', show_time=False)
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                    self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                    mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                        self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                        mode='fan_in', nonlinearity='relu')
    def forward(self, x1, x2):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = x1.shape

        x ,edge= self.backbone(x1, x2)
        out = self.decode_head.forward(x)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        out=out+edge[0]
        if self.aux_head and self.training:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
            return out, aux_fm
        return out,edge[1]

