from networks.decode_head import BaseDecodeHead
from mmseg.ops import resize
from mmseg.models.utils import *
from mmcv.cnn import ConvModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


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
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4_class=_c4
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3_class=_c3
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2_class=_c2
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c1_class=_c1


        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        # class_1234 = _c #self.linear_fuse(torch.cat([_c4_class, _c3_class, _c2_class, _c1_class], dim=1))    


        x = self.dropout(_c)
        x = self.linear_pred(x)

        outs_class = []
        batch_size = x.shape[0]
        
        for i in (range(0,self.num_classes)):
            map_feature = self.conv0(x[:,i,...].unsqueeze(1))
            class_1234 = self.leaky_relu(map_feature)
            class_1234= self.dropout(class_1234)

            class_1234 = self.conv1(class_1234)
            class_1234= self.leaky_relu(class_1234)
            class_1234 = self.dropout(class_1234)

            class_1234 = self.avgpool(class_1234)
            class_1234 = class_1234.view(batch_size,-1)
            class_1234 = self.fc1(class_1234)
            class_1234 = self.fc2(class_1234)            

            outs_class.append(class_1234)

        

        outs_class_main = []
        batch_size = x.shape[0]

        c2_ = self.pre_linear_pred(c2)
        for i in (range(0,self.num_classes)):
            map_feature = self.conv0(c2_[:,i,...].unsqueeze(1))
            class_1234 = self.leaky_relu(map_feature)
            class_1234= self.dropout(class_1234)

            # class_1234 = self.conv1(class_1234)
            # class_1234= self.leaky_relu(class_1234)
            # class_1234 = self.dropout(class_1234)

            class_1234 = self.avgpool(class_1234)
            class_1234 = class_1234.view(batch_size,-1)
            class_1234 = self.fc3(class_1234)
            class_1234 = self.fc4(class_1234)            

            outs_class_main.append(class_1234)

        return x,outs_class,outs_class_main