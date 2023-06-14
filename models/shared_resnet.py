from __future__ import annotations

from typing import List, Type
import torch
from torch import nn, Tensor
from config import Model
from .resnet import BasicBlock, BottleneckBlock
from models.model_utils import conv3x3


class SharedResNet(nn.Module):
    block_dims = [64, 128, 256, 512]

    def __init__(self, num_classes: int, block: Type[BasicBlock] | Type[BottleneckBlock],
                 num_blocks_list: List[int], init_weights: bool = True):
        """
        num_classes: num of classes, e.g., 10 for CIFAR10
        block: block to build ResNet, either BasicBlock or BottleneckBlock
        num_blocks_list: how many blocks for each group
        init_weights: whether to initialize the weights using Kaiming Normal
        """

        super().__init__()

        in_dim_map = {
            "red": 1,
            "red_green": 2,
            "green_blue": 2
        }

        self.first_layers = nn.ModuleDict()

        for data_channel in in_dim_map:
            conv1_x = nn.Sequential(
                conv3x3(in_dim_map[data_channel], self.block_dims[0], stride=1, padding=1),
                nn.BatchNorm2d(self.block_dims[0]),
                nn.ReLU())

            self.first_layers.add_module(data_channel, conv1_x)

        ## helper for `_make_layer()` to keep track of the current sim_scores dimension for creating the next layer
        self.in_dim = self.block_dims[0]

        ## For conv2_x, set first_stride to 1. For the rest of the groups, set to 2
        self.conv2_x = self._make_group(block, self.block_dims[0], num_blocks_list[0], first_stride=1)
        self.conv3_x = self._make_group(block, self.block_dims[1], num_blocks_list[1], first_stride=2)
        self.conv4_x = self._make_group(block, self.block_dims[2], num_blocks_list[2], first_stride=2)
        self.conv5_x = self._make_group(block, self.block_dims[3], num_blocks_list[3], first_stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.block_dims[-1] * block.expansion, num_classes)

        if init_weights:
            self._initialize_weights()

    def _freeze_all_first_layers(self):
        for layer in self.first_layers.values():
            for l in layer.modules():
                if hasattr(l, "weight"):
                    l.weight.requires_grad = False
                    l.weight.grad = None

    def _unfreeze_all_first_layers(self):
        for layer in self.first_layers.values():
            for l in layer.modules():
                if hasattr(l, "weight"):
                    l.weight.requires_grad = True

    @staticmethod
    def _unfreeze_layer(layer):
        for l in layer.modules():
            if hasattr(l, "weight"):
                l.weight.requires_grad = True

    def _make_group(self, block: Type[BasicBlock] | Type[BottleneckBlock], out_dim: int, num_blocks: int,
                    first_stride: int) -> nn.Sequential:
        layer_list = []

        ## The first block can have first_stride is either 1 or 2, while other blocks can only have first_stride=1
        strides = [first_stride] + [1] * (num_blocks - 1)

        ## append the rest of blocks, first_stride=1
        for i in range(num_blocks):
            layer_list.append(block(in_dim=self.in_dim, out_dim=out_dim, stride=strides[i]))

            ## update self.in_dim
            self.in_dim = out_dim * block.expansion

        layer = nn.Sequential(*layer_list)
        return layer

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, data_channel: str, freeze_other: bool = False) -> Tensor:
        assert data_channel in self.first_layers, f"data_channel={data_channel} is not valid!"
        conv1 = self.first_layers[data_channel]

        if freeze_other:
            self._freeze_all_first_layers()
            self._unfreeze_layer(conv1)

        out = conv1(x)

        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)

        return out


def sharedresnet18(cfg: Model) -> SharedResNet:
    return SharedResNet(num_classes=cfg.num_classes, block=BasicBlock,
                        num_blocks_list=[2, 2, 2, 2], init_weights=cfg.init_weights)
