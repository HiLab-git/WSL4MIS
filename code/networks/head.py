from networks.decode_head import BaseDecodeHead
from mmseg.ops import resize
from mmseg.models.utils import *
from mmcv.cnn import ConvModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.distributions.uniform import Uniform
import numpy as np

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
            # norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.pre_linear_pred = nn.Conv2d(64, self.num_classes, kernel_size=1)
        # add classifiation
        self.conv0 = nn.Conv2d(1, embedding_dim, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(embedding_dim, embedding_dim*4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(embedding_dim*4, embedding_dim*8, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(embedding_dim*8, embedding_dim*16, kernel_size=4, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d((16, 16))
        # self.avgpool = nn.AvgPool3d((5, 7, 7))
        # self.64vgpool = nn.AvgPool3d((5, 16, 16))
        self.fc1 = nn.Linear(embedding_dim*4, 512)
        self.fc2 = nn.Linear(512, self.num_classes)

        self.fc3 = nn.Linear(embedding_dim, 128)
        self.fc4 = nn.Linear(128, self.num_classes)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        self.Softmax = nn.Softmax()
        self.out = nn.Conv2d(embedding_dim*8,self.num_classes,kernel_size=1)

         



    def forward(self, inputs):
        c1, c2, c3, c4 = inputs # [bz,64,128,128], [bz,128,64,64], [bz,320,32,32], [bz,512,32,32]

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)  # [bz,256,128,128]

        _c3_ = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3_, size=c1.size()[2:],mode='bilinear',align_corners=False)  # [bz,256,128,128]

        _c2_ = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2_, size=c1.size()[2:],mode='bilinear',align_corners=False)  # [bz,256,128,128]

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])  # [bz,256,128,128]

        logit = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))  # [bz,256,128,128]

        x = self.dropout(logit)
        x = self.linear_pred(x)  # [bz,21,128,128]

        x_16 = F.interpolate(x, size=[8,8],mode='bilinear',align_corners=False)  
        x_32 = F.interpolate(x, size=[16,16],mode='bilinear',align_corners=False)  
        x_64 = F.interpolate(x, size=[32,32],mode='bilinear',align_corners=False)  

        return x, x_16, x_32, x_64

        # _c3_ = self.linear_pred(_c3_)
        # _c2_ = self.linear_pred(_c2_)
        # return x,x_16,_c3_,_c2_






class class_Head(BaseDecodeHead):

    def __init__(self, feature_strides, **kwargs):
        super(class_Head, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
            # norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.pre_linear_pred = nn.Conv2d(64, self.num_classes, kernel_size=1)
        # add classifiation
        self.conv0 = nn.Conv2d(256, embedding_dim, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(embedding_dim, embedding_dim*4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(embedding_dim*4, embedding_dim*8, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(embedding_dim*8, embedding_dim*16, kernel_size=4, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d((4, 4))
        # self.avgpool = nn.AvgPool3d((5, 7, 7))
        # self.64vgpool = nn.AvgPool3d((5, 16, 16))
        self.fc1 = nn.Linear(embedding_dim*4, 512)
        self.fc2 = nn.Linear(512, self.num_classes)

        self.fc3 = nn.Linear(embedding_dim, 128)
        self.fc4 = nn.Linear(128, self.num_classes)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        self.Softmax = nn.Softmax()
        self.out = nn.Conv2d(embedding_dim*8,self.num_classes,kernel_size=1)

         



    def forward(self, inputs):
        # x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32


        outs_class = []
        batch_size = inputs.shape[0]
        
        # for i in (range(0,self.num_classes)):
        map_feature = self.conv0(inputs)
        
        class_1234 = self.leaky_relu(map_feature)
        class_1234= self.dropout(class_1234)

        class_1234 = self.conv1(class_1234)
        class_1234= self.leaky_relu(class_1234)
        class_1234 = self.dropout(class_1234)

        class_1234 = self.avgpool(class_1234)
        class_1234 = class_1234.view(batch_size,-1)
        class_1234 = self.fc1(class_1234)
        class_1234 = self.fc2(class_1234)            

        outs_class=class_1234

        


        return outs_class    
    


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)

class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpBlock1(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock1, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        # x1 = self.up(x1)
        # x2 = self.up(x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Unet_Decoder(nn.Module):
    def __init__(self, params):
        super(Unet_Decoder, self).__init__()
        self.params = params
        
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 4)

        # self.up1 = UpBlock(
        #     self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock1(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,kernel_size=3, padding=1)

        # self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,kernel_size=3, padding=1)
        # self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[1], self.n_class,kernel_size=3, padding=1)
        self.feature_noise = FeatureNoise()


        self.out_conv_x23= nn.Conv2d(self.ft_chns[3]*2, self.ft_chns[3],kernel_size=1, padding=1)

    def forward(self, feature,shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x = feature[3]



        x = self.up2(x, x2)
        if self.training:
            dp2_out_seg = self.out_conv_dp2(Dropout(x, p=0.5))
        else:
            dp2_out_seg = self.out_conv_dp2(x)
        # dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        if self.training:
            dp3_out_seg = self.out_conv_dp3(FeatureDropout(x))
        else:
            dp3_out_seg = self.out_conv_dp3(x)
        # dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)





        
        return dp0_out_seg, dp2_out_seg, dp3_out_seg


        # x = self.up2(x, x2)
        # x = self.up3(x, x1)
        # x = self.up4(x, x0)
        # output = self.out_conv(x)
        # return output




def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x