import numpy as np

np.set_printoptions(threshold=np.inf)

import network.scribbleVC_Prostate
class scribbleVC_Prostate(network.scribbleVC_Prostate.Net):
    def __init__(self, linear_layer, bilinear, batch_size=None, num_classes=3):
        super().__init__(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
                num_heads=6, mlp_ratio=4, qkv_bias=True, drop_rate=0.0, drop_path_rate=0.1,
                in_chans=1, num_classes=num_classes, linear_layer=linear_layer, bilinear=bilinear, batch_size=batch_size)

import network.scribbleVC_ACDC
class scribbleVC_ACDC(network.scribbleVC_ACDC.Net):
    def __init__(self, linear_layer, bilinear, batch_size=None, num_classes=4):
        super().__init__(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
                num_heads=6, mlp_ratio=4, qkv_bias=True, drop_rate=0.0, drop_path_rate=0.1,
                in_chans=1, num_classes=num_classes, linear_layer=linear_layer, bilinear=bilinear, batch_size=batch_size)
