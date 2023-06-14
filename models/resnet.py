## Paper: Deep Residual Learning for Image Recognition - https://arxiv.org/pdf/1512.03385v1.pdf

## references:
# [1] https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
# [2] https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
# [3] Pytorch: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

##### Resnet-N, where N is the depth of the network
# ResNet consists of
# 1 conv stem -> 4 groups -> 1 FC.

# Each group consists of some blocks and 1 shortcut (optional).
# Each block has 2 (BasicBlock) or 3 (BottleneckBlock) conv layers
# Depth of ResNet18: 1 conv stem + (2+2+2+2) * 2 + 1 FC = 18 layers. Note this doesn't count all conv layers, such as the 1x1 convs in the shortcut.

from __future__ import annotations
from typing import List, Type
import torch
from torch import nn, Tensor

from config import Model
from models.model_utils import conv3x3, conv1x1
from utils import analyze_model


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_dim: int, out_dim: int, stride: int):
        super().__init__()

        ## For sequence, we have Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm
        self.sequence = nn.Sequential(
            conv3x3(in_dim, out_dim, stride=stride, padding=1), ## only downsample here if any
            nn.BatchNorm2d(out_dim, momentum=0.1),
            nn.ReLU(),

            conv3x3(out_dim, out_dim * self.expansion, stride=1, padding=1), ## set first_stride to 1
            nn.BatchNorm2d(out_dim * self.expansion, momentum=0.1)
        )

        ## Shortcut for residual connection
        self.shortcut = nn.Sequential()
        # if first_stride != 1 or the size of input changes after going through `sequence`,
        # we'll use 1x1 Conv to match the dimension
        if stride != 1 or in_dim != out_dim * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(in_dim, out_dim * self.expansion, stride=stride),
                nn.BatchNorm2d(out_dim * self.expansion, momentum=0.1))

        ## Last activation
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.sequence(x) + self.shortcut(x)
        out = self.relu(out)
        return out


class BottleneckBlock(nn.Module):
    """
    Similar to BasicBlock above, only difference in the `sequence`:
     conv1x1 -> conv3x3 -> conv1x1, conv3x3 is the bottleneck in terms of spatial dimensions.
    """
    expansion = 4

    def __init__(self, in_dim: int, out_dim: int, stride: int):
        super().__init__()

        self.sequence = nn.Sequential(
            conv1x1(in_dim, out_dim),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),

            conv3x3(out_dim, out_dim, stride=stride, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),

            conv1x1(out_dim, out_dim * self.expansion),
            nn.BatchNorm2d(out_dim * self.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_dim != out_dim * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(in_dim, out_dim * self.expansion, stride=stride),
                nn.BatchNorm2d(out_dim * self.expansion)
            )

        self.relu = nn.ReLU()

    def forward(self, x: Tensor):
        out = self.sequence(x) + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    Paper: Deep Residual Learning for Image Recognition - https://arxiv.org/pdf/1512.03385v1.pdf
    """
    group_dims = [64, 128, 256, 512]

    def __init__(self, input_dim, num_classes: int, block: Type[BasicBlock] | Type[BottleneckBlock],
                 num_blocks_list: List[int], init_weights: bool = True):
        """
        input_dim: num of channels of the input image, (e.g., 3)
        num_classes: num of classes, e.g., 10 for CIFAR10
        block: block to build ResNet, either BasicBlock or BottleneckBlock
        num_blocks_list: how many blocks for each group
        init_weights: whether to initialize the weights using Kaiming Normal
        """

        super().__init__()

        ### based on Table 1, page 5 in the paper
        self.conv1 = nn.Sequential(
            conv3x3(input_dim, self.group_dims[0], stride=1, padding=1),
            nn.BatchNorm2d(self.group_dims[0], momentum=0.1),
            nn.ReLU())

        ## helper for `_make_group()` to keep track of the current sim_scores dimension for creating the next group
        self.in_dim = self.group_dims[0]

        ## For conv2_x, set first_stride to 1. For the rest of the groups, set to 2 to reduce the spatial dimensions by half. Note that from conv3_x group afterwards, there'll be a `shortcut` at the first block.
        self.conv2_x = self._make_group(block, self.group_dims[0], num_blocks_list[0], first_stride=1)
        self.conv3_x = self._make_group(block, self.group_dims[1], num_blocks_list[1], first_stride=2)
        self.conv4_x = self._make_group(block, self.group_dims[2], num_blocks_list[2], first_stride=2)
        self.conv5_x = self._make_group(block, self.group_dims[3], num_blocks_list[3], first_stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.group_dims[-1] * block.expansion, num_classes)

        if init_weights:
            self._initialize_weights()

    def _make_group(self, block: Type[BasicBlock] | Type[BottleneckBlock], out_dim: int, num_blocks: int,
                    first_stride: int) -> nn.Sequential:
        blocks = []

        ## The first block can have first_stride is either 1 or 2, while other blocks can only have first_stride=1
        strides = [first_stride] + [1] * (num_blocks - 1)

        ## append the rest of blocks, first_stride=1
        for i in range(num_blocks):
            blocks.append(block(in_dim=self.in_dim, out_dim=out_dim, stride=strides[i]))
            ## update self.in_dim for the next block, in case block.expansion != 1
            self.in_dim = out_dim * block.expansion
        return nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet18(cfg: Model):
    """
    Depth: 1 conv stem + (2+2+2+2) * 2 + 1 FC = 18 deep layers
    @param cfg:
    @return:
    """
    num_blocks_list = [2, 2, 2, 2]
    return ResNet(cfg.in_dim, cfg.num_classes, block=BasicBlock, num_blocks_list=num_blocks_list,
                  init_weights=cfg.init_weights)


def resnet34(cfg: Model):
    """ return a ResNet 34 object
    """
    num_blocks_list = [3, 4, 6, 3]
    return ResNet(cfg.in_dim, cfg.num_classes, block=BasicBlock, num_blocks_list=num_blocks_list,
                  init_weights=cfg.init_weights)


def resnet50(cfg: Model):
    """ return a ResNet 50 object
    """
    num_blocks_list = [3, 4, 6, 3]
    return ResNet(cfg.in_dim, cfg.num_classes, block=BottleneckBlock, num_blocks_list=num_blocks_list,
                  init_weights=cfg.init_weights)


def resnet101(cfg: Model):
    """ return a ResNet 101 object
    """
    num_blocks_list = [3, 4, 23, 3]
    return ResNet(cfg.in_dim, cfg.num_classes, block=BottleneckBlock, num_blocks_list=num_blocks_list,
                  init_weights=cfg.init_weights)


def resnet152(cfg: Model):
    """ return a ResNet 101 object
    """
    num_blocks_list = [3, 8, 36, 3]
    return ResNet(cfg.in_dim, cfg.num_classes, block=BottleneckBlock, num_blocks_list=num_blocks_list,
                  init_weights=cfg.init_weights)


if __name__ == '__main__':
    model = resnet18(Model(in_dim=3, num_classes=10, init_weights=True, name="resnet18"))
    analyze_model(model)