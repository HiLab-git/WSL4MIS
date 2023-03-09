# coding=utf-8
# This file borrowed from Swin-UNet: https://github.com/HuCaoFighting/Swin-Unet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from mmseg.ops import resize

from networks.mix_transformer import MixVisionTransformer,SegFormerHead
from functools import partial
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
# dict(type='SyncBN', requires_grad=True)

logger = logging.getLogger(__name__)




logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        # self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
        #                         patch_size=config.MODEL.SWIN.PATCH_SIZE,
        #                         in_chans=config.MODEL.SWIN.IN_CHANS,
        #                         num_classes=self.num_classes,
        #                         embed_dim=config.MODEL.SWIN.EMBED_DIM,
        #                         depths=config.MODEL.SWIN.DEPTHS,
        #                         num_heads=config.MODEL.SWIN.NUM_HEADS,
        #                         window_size=config.MODEL.SWIN.WINDOW_SIZE,
        #                         mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        #                         qkv_bias=config.MODEL.SWIN.QKV_BIAS,
        #                         qk_scale=config.MODEL.SWIN.QK_SCALE,
        #                         drop_rate=config.MODEL.DROP_RATE,
        #                         drop_path_rate=config.MODEL.DROP_PATH_RATE,
        #                         ape=config.MODEL.SWIN.APE,
        #                         patch_norm=config.MODEL.SWIN.PATCH_NORM,
        #                         use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        
        self.mix_transformer = MixVisionTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dims=[32, 64, 160, 256],
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=[1, 2, 5, 8],                             
                                mlp_ratios=[4, 4, 4, 4],
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                stride=[4,2,2,1],
                            )
                                
                            
# num_heads=[1, 2, 5, 8], [32, 64, 160, 256],
# config.MODEL.SWIN.EMBED_DIM,
# config.MODEL.SWIN.MLP_RATIO,config.MODEL.SWIN.NUM_HEADS,  
        # self.segformerhead = SegFormerHead(in_channels=,channels=,
        #          num_classes=self.num_classes,
        #          dropout_ratio=0.1,
        #          conv_cfg=None,
        #          norm_cfg=None,
        #          act_cfg=dict(type='ReLU'),
        #          in_index=-1,
        #          input_transform=None,
        #          decoder_params=None,
        #          ignore_index=255,
        #          sampler=None,
        #          align_corners=False)
    # def __init__(self,
    #              in_channels,
    #              channels,
    #              *,
    #              num_classes,
    #              dropout_ratio=0.1,
    #              conv_cfg=None,
    #              norm_cfg=None,
    #              act_cfg=dict(type='ReLU'),
    #              in_index=-1,
    #              input_transform=None,
    #              loss_decode=dict(
    #                  type='CrossEntropyLoss',
    #                  use_sigmoid=False,
    #                  loss_weight=1.0),
    #              decoder_params=None,
    #              ignore_index=255,
    #              sampler=None,
    #              align_corners=False)
        

        self.seg_head=SegFormerHead(
                            # type='SegFormerHead',
                            in_channels=[32, 64, 160, 256],
                            in_index=[0, 1, 2, 3],
                            feature_strides=[4, 8, 16, 32],
                            channels=128,
                            dropout_ratio=0.1,
                            num_classes=4,
                            norm_cfg=norm_cfg,
                            align_corners=False,
                            decoder_params=dict(embed_dim=256),
                            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        



                
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        # logits = self.swin_unet(x)
        logits = self.mix_transformer(x)
        logits=self.seg_head(logits[0])
        out = resize(
            input=logits,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False)
        return out

    # def load_from(self, config):
    #     pretrained_path = config.MODEL.PRETRAIN_CKPT
    #     if pretrained_path is not None:
    #         print("pretrained_path:{}".format(pretrained_path))
    #         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #         pretrained_dict = torch.load(pretrained_path, map_location=device)
    #         if "model"  not in pretrained_dict:
    #             print("---start load pretrained modle by splitting---")
    #             pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
    #             for k in list(pretrained_dict.keys()):
    #                 if "output" in k:
    #                     print("delete key:{}".format(k))
    #                     del pretrained_dict[k]
    #             msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
    #             # print(msg)
    #             return
    #         pretrained_dict = pretrained_dict['model']
    #         print("---start load pretrained modle of swin encoder---")

    #         model_dict = self.swin_unet.state_dict()
    #         full_dict = copy.deepcopy(pretrained_dict)
    #         for k, v in pretrained_dict.items():
    #             if "layers." in k:
    #                 current_layer_num = 3-int(k[7:8])
    #                 current_k = "layers_up." + str(current_layer_num) + k[8:]
    #                 full_dict.update({current_k:v})
    #         for k in list(full_dict.keys()):
    #             if k in model_dict:
    #                 if full_dict[k].shape != model_dict[k].shape:
    #                     print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
    #                     del full_dict[k]

    #         msg = self.swin_unet.load_state_dict(full_dict, strict=False)
    #         # print(msg)
    #     else:
    #         print("none pretrain")
 