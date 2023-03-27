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
from networks.head import SegFormerHead,Unet_Decoder,class_Head
from functools import partial
import pickle
from utils.util import FeatureDropout

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
                                patch_size=4,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dims=[32, 64, 160, 256],
                                depths=[2, 2, 2, 2],
                                num_heads=[1, 2, 5, 8],                             
                                mlp_ratios=[4, 4, 4, 4],
                                qkv_bias=True,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                stride=[4,2,2,1],
                            )
# class mit_b0(MixVisionTransformer):
#     def __init__(self, stride=None, **kwargs):
#         super(mit_b0, self).__init__(
#             patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1, stride=stride)        

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
        
        params = {
                  'feature_chns': [32, 64, 160, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3],
                  'class_num': 4,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.unet_decoder = Unet_Decoder(params)



        self.calss_head=class_Head(
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


                
    def forward(self, x,aux=False):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        # pickle.load = partial(pickle.load, encoding="latin1")
        # pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        
        
        
        state_dict = torch.load('/mnt/sdd/tb/pretrained/'+'mit_b0'+'.pth')
        state_dict.pop('head.weight')
        state_dict.pop('head.bias')
        self.mix_transformer.load_state_dict(state_dict,)
        
        logits = self.mix_transformer(x)

        _attns =logits[1]

        # attn_cat  = torch.cat(_attns[3], dim=1)#.detach()
        # attn_cat  = attn_cat + attn_cat.permute(0, 1, 3, 2)
        # _attns = self.attn_proj(attn_cat)
        # _attns = torch.sigmoid(_attns)#[:,0,...]

        attn_cat3 = torch.cat([_attns[3][0],_attns[3][1]] ,dim=1)#.detach()
        attn_cat3 = attn_cat3 + attn_cat3.permute(0, 1, 3, 2)
        attn_pred3 = self.attn_proj(attn_cat3)
        attn_pred3 = torch.sigmoid(attn_pred3)[:,0,...]






        mlp_f=self.seg_head(logits[0])
        # calss=self.calss_head(logits[0][3])
        if aux: 
            return mlp_f ,attn_pred3,_attns
        # aux3_feature = [FeatureDropout(i) for i in logits[0]]
        else:
            shape = x.shape[2:] 
            dp0_out_seg,dp2_out_seg,dp3_out_seg=self.unet_decoder(logits[0],shape)
            # mlp_seg_aux3=self.seg_head(aux3_feature)
            
            
            
            # mlp_seg = F.interpolate(input=mlp_seg,size=x.shape[2:], mode='bilinear',align_corners=False)    

            
                # logits_unethead=self.unet_decoder(logits[0],shape)
            logits_unethead = F.interpolate(input=dp0_out_seg,size=x.shape[2:], mode='bilinear',align_corners=False)



            return logits_unethead,mlp_f,attn_pred3,_attns





 