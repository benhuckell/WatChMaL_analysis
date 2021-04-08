import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  

from watchmal.utils.frrn_utils import FRRU, RU, conv3DBatchNormRelu, conv3DGroupNormRelu

frrn_specs_dic = {
    "A": {
        "encoder": [[3, 8, (1,2,2)]],
        "decoder": [],
    },
    "B": {
        "encoder": [[3, 8, (1,2,2)], [4, 16, (1,4,4)], [2, 32, (1,8,8)]],
        "decoder": [[2, 16, (1,4,4)], [2, 4, (1,2,2)]],
    },
    "C": {
        "encoder": [[3, 16, (1,2,2)], [4, 32, (1,4,4)], [2, 64, (1,8,8)]],
        "decoder": [[2, 32, (1,4,4)], [2, 8, (1,2,2)]],
    }
}


class frrn(nn.Module):


    def __init__(self, n_classes=4, model_type="B", group_norm=False, n_groups=16):

        super(frrn, self).__init__()
        self.n_channels = 4
        self.n_classes = n_classes
        self.model_type = model_type
        self.group_norm = group_norm
        self.n_groups = n_groups

        if self.group_norm:
            self.conv1 = conv3DBatchNormRelu(1, self.n_channels, (19,3,3), 1, (9,1,1))
        else:
            self.conv1 = conv3DBatchNormRelu(1, self.n_channels, (19,3,3), 1, (9,1,1))

        self.up_residual_units = []
        self.down_residual_units = []
        for i in range(3):
            self.up_residual_units.append(
                RU(
                    channels=self.n_channels,
                    kernel_size=(19,3,3),
                    padding = (9,1,1),
                    strides=1,
                    group_norm=self.group_norm,
                    n_groups=self.n_groups,
                )
            )
            self.down_residual_units.append(
                RU(
                    channels=self.n_channels,
                    kernel_size=(19,3,3),
                    padding = (9,1,1),
                    strides=1,
                    group_norm=self.group_norm,
                    n_groups=self.n_groups,
                )
            )

        self.up_residual_units = nn.ModuleList(self.up_residual_units)
        self.down_residual_units = nn.ModuleList(self.down_residual_units)

        self.split_conv_3d = nn.Conv3d(self.n_channels, self.n_channels, kernel_size=(19,1,1), padding=(9,0,0), stride=1, bias=False)

        
        # each spec is as (n_blocks, channels, scale)
        self.encoder_frru_specs = frrn_specs_dic[self.model_type]["encoder"]

        self.decoder_frru_specs = frrn_specs_dic[self.model_type]["decoder"]
        
        
        # encoding
        prev_channels = self.n_channels
        self.encoding_frrus = {}
        for n_blocks, channels, scale in self.encoder_frru_specs:
            for block in range(n_blocks):
                key = "_".join(map(str, ["encoding_frru", n_blocks, channels, scale, block]))
                setattr(
                    self,
                    key,
                    FRRU(
                        prev_channels=prev_channels,
                        out_channels=channels,
                        scale=scale,
                        group_norm=self.group_norm,
                        n_groups=self.n_groups,
                    ),
                )
            prev_channels = channels

        # decoding
        self.decoding_frrus = {}
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = "_".join(map(str, ["decoding_frru", n_blocks, channels, scale, block]))
                setattr(
                    self,
                    key,
                    FRRU(
                        prev_channels=prev_channels,
                        out_channels=channels,
                        scale=scale,
                        group_norm=self.group_norm,
                        n_groups=self.n_groups,
                    ),
                )
            prev_channels = channels
        
        self.merge_conv_3d = nn.Conv3d(
            prev_channels + self.n_channels, self.n_channels, kernel_size=(19,1,1), padding=(9,0,0), stride=1, bias=False
        )

        self.classif_conv_3d = nn.Conv3d(
            self.n_channels, self.n_classes, kernel_size=(19,1,1), padding=(9,0,0), stride=1, bias=True
        )

        self.activationFunction = nn.ReLU(inplace=False)

    def forward(self, x):

        # pass to initial conv
        x = self.conv1(x)


        # pass through residual units
        for i in range(3):
            x = self.up_residual_units[i](x)

        # divide stream
        y = x
        z = self.split_conv_3d(x)

        #print("Model Shape:", z.shape)
        #print("Model Shape x:", x.shape)

        prev_channels = self.n_channels

        
        # encoding
        for n_blocks, channels, scale in self.encoder_frru_specs:
            #print("Settings:",channels, scale)
            # maxpool bigger feature map
            y_pooled = F.max_pool3d(y, stride=(1,2,2), kernel_size=(1,2,2), padding=0)
            # pass through encoding FRRUs
            for block in range(n_blocks):
                key = "_".join(map(str, ["encoding_frru", n_blocks, channels, scale, block]))
                y, z = getattr(self, key)(y_pooled, z)
            prev_channels = channels

        # decoding
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # bilinear upsample smaller feature map
            upsample_size = torch.Size([dimShape * dimScale for dimShape, dimScale in zip(y.shape[2:],[1,2,2])])
            y_upsampled = F.upsample(y, size=upsample_size, mode="trilinear", align_corners=True)

            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = "_".join(map(str, ["decoding_frru", n_blocks, channels, scale, block]))
                # print("Incoming FRRU Size: ", key, y_upsampled.shape, z.shape)
                y, z = getattr(self, key)(y_upsampled, z)
                # print("Outgoing FRRU Size: ", key, y.shape, z.shape)
            prev_channels = channels
        
        # merge streams
        x = torch.cat(
            [F.upsample(y, scale_factor=(1,2,2), mode="trilinear", align_corners=True), z], dim=1
        )
        x = self.merge_conv_3d(x)

        # pass through residual units
        for i in range(3):
            x = self.down_residual_units[i](x)
        
        # final 1x1 conv to get classification
        x = self.classif_conv_3d(x)

        #Activation function - ReLU
        x = self.activationFunction(x)
        

        return x
