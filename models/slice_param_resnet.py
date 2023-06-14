from __future__ import annotations

from typing import List, Type, Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from config import Model
from helper_classes.norm_type import NormType
from models.resnet import BasicBlock, BottleneckBlock
from einops import repeat


class SliceParamResNet(nn.Module):
    block_dims = [64, 128, 256, 512]

    def __init__(self, num_classes: int, block: Type[BasicBlock] | Type[BottleneckBlock],
                 num_blocks_list: List[int], in_channel_names: List[str],
                 separate_norm: Optional[bool], norm_type: NormType,
                 image_h_w: Optional[List[int]], duplicate: bool, init_weights: bool = True):
        """
        num_classes: num of classes, e.g., 10 for CIFAR10
        block: block to build ResNet, either BasicBlock or BottleneckBlock
        num_blocks_list: how many blocks for each group
        in_channel_names: List of input channels, e.g., ["red", "green", "blue"]. Note that the order matters. We use this order to slice the params of the first layer.
        separate_norm: whether to use separate norm layer for each data chunk
        norm_type: type of normalization layer, e.g., NormType.BATCH_NORM
        image_h_w: input height/width, e.g., 32 for CIFAR10. Used when norm_type=True
        duplicate: where to only use the first param bank and duplicate for all the channels
        init_weights: whether to initialize the weights using Kaiming Normal
        """

        super().__init__()

        self.duplicate = duplicate
        if self.duplicate:
            total_in_channels = 1
        else:
            total_in_channels = len(in_channel_names)
        self.separate_norm = separate_norm
        self.norm_type = norm_type

        self.mapper = {
            "red": [0],
            "red_green": [0, 1],
            "green_blue": [1, 2]
        }
        # First conv layer
        self.conv1_param_bank = nn.Parameter(torch.zeros(self.block_dims[0], total_in_channels, 3, 3))
        if norm_type == NormType.LAYER_NORM:
            NormClass = nn.LayerNorm
            norm_param = [self.block_dims[0], *image_h_w]
        elif norm_type == NormType.BATCH_NORM:
            NormClass = nn.BatchNorm2d
            norm_param = self.block_dims[0]
        elif norm_type == NormType.INSTANCE_NORM:
            NormClass = nn.InstanceNorm2d
            norm_param = self.block_dims[0]
        else:
            raise ValueError(f"norm_type {norm_type} not supported")

        if self.separate_norm:
            self.norm_1_dict = nn.ModuleDict({
                data_channel: NormClass(norm_param) for data_channel in self.mapper
            })
        else:
            self.norm_1 = NormClass(norm_param)

        self.relu1 = nn.ReLU()

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
        nn.init.kaiming_normal_(self.conv1_param_bank, mode="fan_in", nonlinearity="relu")

    def slice_params_first_layer(self, data_channel: str) -> Tensor:
        assert data_channel in self.mapper, f"Invalid data_channel: {data_channel}"
        if self.duplicate:
            ## conv1depthwise_param_bank's shape: (out_dim, 1, 3, 3)
            params = repeat(self.conv1_param_bank, "o i h w -> o (i c) h w", c=len(self.mapper[data_channel]))
        else:
            params = self.conv1_param_bank[:, self.mapper[data_channel]]
        return params

    def forward(self, x: Tensor, data_channel: str) -> Tensor:
        ## slice params of the first layer:
        conv1_params = self.slice_params_first_layer(data_channel)
        out = F.conv2d(x, conv1_params, bias=None, stride=1, padding=1)

        if self.separate_norm:
            out = self.norm_1_dict[data_channel](out)
        else:
            out = self.norm_1(out)
        out = self.relu1(out)

        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)

        return out


def sliceparamresnet18(cfg: Model) -> SliceParamResNet:
    return SliceParamResNet(num_classes=cfg.num_classes, block=BasicBlock,
                            num_blocks_list=[2, 2, 2, 2], in_channel_names=cfg.in_channel_names,
                            init_weights=cfg.init_weights, separate_norm=cfg.separate_norm,
                            norm_type=cfg.norm_type, image_h_w=getattr(cfg, "image_h_w", None),
                            duplicate=cfg.duplicate)

# if __name__ == '__main__':
#     model = sliceparamresnet18(
#         Model(name="sliceparamresnet18", num_classes=10, in_channel_names=["red", "green", "blue"], init_weights=True, separate_norm=True))
#
#     print(model)
