import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import torchvision.ops as ops
from einops import rearrange
from einops.layers.torch import Rearrange

import copy

class synthnet(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.past_length = config.past_length
        self.patch_size = config.patch_size if "patch_size" in config else 4
        input_dim = int(4*(self.past_length+1)*(self.patch_size**2))
        output_dim = int(4*(self.patch_size**2))
        self.patchify = nn.Sequential(Rearrange("b c (h p1) (w p2) -> b (p1 p2 c) h w",p1 =self.patch_size,p2 = self.patch_size ),
                             nn.Conv2d(input_dim, input_dim//2, kernel_size=1, stride=1))
        self.body = SimpleUnet(input_dim//2,output_dim)
        self.head = nn.Sequential(
                            nn.Conv2d(output_dim, output_dim, kernel_size=1, stride=1, bias=False),
                            nn.ReLU(),
                            nn.Conv2d(output_dim, output_dim, kernel_size=1, stride=1 ,bias=False),
                            Rearrange("b (p1 p2 c) h w -> b c (h p1) (w p2)",p1 =self.patch_size,p2 = self.patch_size )
                        )
        
        
    
    def forward(self,warped_input):
        x = self.patchify(warped_input)
        x = self.body(x)
        x = self.head(x).contiguous()
        outputs = dict()
        outputs["psuedo_img"] = x[:,:3]
        outputs["psuedo_depth"] = F.relu(x[:,3:4])
        return outputs



def convbn_2d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                padding=pad, bias=False),
        nn.BatchNorm2d( out_channels)
    )


class SimpleUnet(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(SimpleUnet, self).__init__()

        self.conv1 = nn.Sequential(
            convbn_2d(in_channels, in_channels * 2, 3, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            convbn_2d(in_channels * 2, in_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            convbn_2d(in_channels * 2, in_channels * 4, 3, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            convbn_2d(in_channels * 4, in_channels * 4, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2)
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.redir1 = convbn_2d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_2d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return self.out(conv6) 