from __future__ import annotations

from typing import List, Type, Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from config import Model
from helper_classes.norm_type import NormType
from models.resnet import BasicBlock, BottleneckBlock
from einops import rearrange


class HyperNetwork(nn.Module):

    def __init__(self, z_dim, d, kernel_size, out_size, in_size=1):
        super().__init__()
        self.z_dim = z_dim
        self.d = d  ## in the paper, d = z_dim
        self.kernel_size = kernel_size
        self.out_size = out_size
        self.in_size = in_size

        self.W = nn.Parameter(torch.randn((self.z_dim, self.in_size, self.d)))
        self.b = nn.Parameter(torch.randn((self.in_size, self.d)))

        self.W_out = nn.Parameter(torch.randn((self.d, self.out_size, self.kernel_size, self.kernel_size)))
        self.b_out = nn.Parameter(torch.randn((self.out_size, self.kernel_size, self.kernel_size)))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.W)
        nn.init.kaiming_normal_(self.W_out)

    def forward(self, z: Tensor) -> Tensor:
        """
        @param z: (num_channels, z_dim)
        @return: kernel (out_size, in_size, kernel_size, kernel_size)
        """
        a = torch.einsum('c z, z i d ->c i d', z, self.W) + self.b
        K = torch.einsum('c i d, d o h w ->c i o h w', a, self.W_out) + self.b_out
        K = rearrange(K, 'c i o h w -> o (c i) h w')
        return K


class HyperNetworkResNet(nn.Module):
    block_dims = [64, 128, 256, 512]

    def __init__(self, num_classes: int, block: Type[BasicBlock] | Type[BottleneckBlock],
                 num_blocks_list: List[int], in_channel_names: List[str],
                 separate_norm: Optional[bool], norm_type: NormType,
                 image_h_w: Optional[List[int]], z_dim: int, hidden_dim: int,
                 separate_emb: bool, init_weights: bool = True):
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

        total_in_channels = len(in_channel_names)
        self.separate_norm = separate_norm
        self.norm_type = norm_type
        self.separate_emb = separate_emb

        self.mapper = {
            "red": [0],
            "red_green": [0, 1],
            "green_blue": [1, 2]
        }
        # First conv layer
        if self.separate_emb:
            self.conv1_emb = nn.ParameterDict({
                data_channel: torch.randn(len(channels), z_dim) for data_channel, channels in self.mapper.items()
            })
        else:
            self.conv1_emb = nn.Embedding(total_in_channels, z_dim)

        self.hypernet = HyperNetwork(z_dim, hidden_dim, 3, self.block_dims[0], 1)

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

    def generate_params_first_layer(self, data_channel: str) -> Tensor:
        assert data_channel in self.mapper, f"Invalid data_channel: {data_channel}"
        if self.separate_emb:
            z_emb = self.conv1_emb[data_channel]
        else:
            z_emb = self.conv1_emb(torch.tensor(self.mapper[data_channel], dtype=torch.long, device=self.conv1_emb.weight.device))

        kernels = self.hypernet(z_emb)
        return kernels

    def forward(self, x: Tensor, data_channel: str) -> Tensor:
        ## slice params of the first layer:
        conv1_params = self.generate_params_first_layer(data_channel)
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


def hypernetwork_resnet18(cfg: Model) -> HyperNetworkResNet:
    return HyperNetworkResNet(num_classes=cfg.num_classes, block=BasicBlock,
                              num_blocks_list=[2, 2, 2, 2], in_channel_names=cfg.in_channel_names,
                              init_weights=cfg.init_weights, separate_norm=cfg.separate_norm,
                              norm_type=cfg.norm_type, image_h_w=getattr(cfg, "image_h_w", None),
                              z_dim=cfg.z_dim, hidden_dim=cfg.hidden_dim,
                              separate_emb=cfg.separate_emb)
