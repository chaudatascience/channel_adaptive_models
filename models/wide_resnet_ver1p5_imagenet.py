
## ResNet50 with wide option for ImageNet
# This implementation is based on ResNet50 v1.5 (first_stride 2 on 3x3 convolutions).
# Different from v1, in the bottleneck blocks,
# while v1 has a first_stride of 2 in the first 1x1 conv to downsample,
# v1.5 has first_stride = 2 in the 3x3 conv to do the job.
# In contrast to the WRN paper, this applies batch norm, relu after convolution.
# https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/resources/resnet_pyt

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class WideResNet(nn.Module):
    def __init__(self):
        super(WideResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.block1 = self._make_block(16, 16, 2, stride=1)
        self.block2 = self._make_block(16, 32, 2, stride=2)
        self.block3 = self._make_block(32, 64, 2, stride=2)

        self.linear = nn.Linear(64, 10)
