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
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from mmseg.ops import resize

from networks.mix_transformer import MixVisionTransformer
from functools import partial
from networks.head import SegFormerHead
from functools import partial
import pickle
# from networks.decode_head import Classification_head
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
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
        
        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)        
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")    

        self.classifier = nn.Conv2d(in_channels=256, out_channels=self.num_classes-1, kernel_size=1, bias=False)    


                
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        # pickle.load = partial(pickle.load, encoding="latin1")
        # pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        # state_dict = torch.load('/mnt/sdd/yd2tb/pretrained/'+'mit_b0'+'.pth', map_location=lambda storage, loc: storage, pickle_module=pickle)            
        state_dict = torch.load('/mnt/sdd/yd2tb/pretrained/'+'mit_b0'+'.pth')
        state_dict.pop('head.weight')
        state_dict.pop('head.bias')
        self.mix_transformer.load_state_dict(state_dict,)
        
        logits = self.mix_transformer(x)
        _attns =logits[1]
        attn_cat  = torch.cat(_attns[-2:], dim=1)#.detach()
        attn_cat  = attn_cat + attn_cat.permute(0, 1, 3, 2)
        attn_pred = self.attn_proj(attn_cat)
        attn_pred = torch.sigmoid(attn_pred)#[:,0,...]



        logits_=self.seg_head(logits[0])
        calss=logits_[1]

        out=F.interpolate(logits_[0], size=x.shape[2:], mode='bilinear', align_corners=False)
        # out = resize(input=logits_[0],ize=x.shape[2:], mode='bilinear',align_corners=False)

        # affinity_ = torch.softmax(torch.matmul(attn_pred, torch.softmax(out, dim=-1)),dim=-1)   

        return out,calss,attn_pred





 