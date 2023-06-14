from typing import Optional

import numpy as np
from einops import rearrange
from timm import create_model
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from config import Model, AttentionPoolingParams
from helper_classes.channel_pooling_type import ChannelPoolingType
from helper_classes.feature_pooling import FeaturePooling
from models.channel_attention_pooling import ChannelAttentionPoolingLayer
from models.model_utils import conv1x1


# model here: https://github.com/huggingface/pytorch-image-models/blob/b3e816d6d71ec132b39c603d68b619ae2870fd0a/timm/models/convnext.py#L410
# lr=4.0e-3 (mentioned in  A ConvNet for the 2020s paper)


class DepthwiseConvNeXt(nn.Module):
    def __init__(
        self,
        config: Model,
        attn_pooling_params: Optional[AttentionPoolingParams] = None,
    ):
        # pretrained_model_name "convnext_tiny.fb_in22k"
        ## forward pass: https://github.com/huggingface/pytorch-image-models/blob/b3e816d6d71ec132b39c603d68b619ae2870fd0a/timm/models/convnext.py#L420

        super().__init__()
        self.cfg = config
        model = create_model(config.pretrained_model_name, pretrained=config.pretrained)

        self.kernels_per_channel = config.kernels_per_channel
        self.pooling_channel_type = config.pooling_channel_type

        ## all channels in this order (alphabet): ['er', 'golgi', 'membrane', 'microtubules','mito','nucleus','protein', 'rna']
        self.mapper = {
            "Allen": [5, 2, 6],
            "HPA": [3, 6, 5, 0],
            "CP": [5, 0, 7, 1, 4],
        }

        out_dim, original_in_dim, kh, kw = model.stem[0].weight.shape
        self.stride = model.stem[0].stride
        self.padding = model.stem[0].padding
        self.dilation = model.stem[0].dilation
        self.groups = model.stem[0].groups

        total_in_channels = len(config.in_channel_names)

        self.get_patch_emb = nn.ModuleDict()
        patch = 4
        for chunk in self.mapper:
            self.get_patch_emb[chunk] = nn.Conv2d(
                len(self.mapper[chunk]),
                len(self.mapper[chunk]),
                kernel_size=patch,
                stride=patch,
                padding=0,
                groups=len(self.mapper[chunk]),
            )

        self.conv1depthwise_param_bank = nn.Parameter(
            torch.zeros(total_in_channels * self.kernels_per_channel, 1, 3, 3)
        )

        self.conv_1x1 = conv1x1(self.kernels_per_channel, out_dim)

        if self.pooling_channel_type in [
            ChannelPoolingType.WEIGHTED_SUM_RANDOM,
            ChannelPoolingType.WEIGHTED_SUM_RANDOM_NO_SOFTMAX,
        ]:
            self.weighted_sum_pooling = nn.Parameter(torch.randn(total_in_channels))

        if self.pooling_channel_type in [
            ChannelPoolingType.WEIGHTED_SUM_RANDOM_PAIRWISE_NO_SOFTMAX,
            ChannelPoolingType.WEIGHTED_SUM_RANDOM_PAIRWISE,
        ]:
            self.weighted_sum_pooling = torch.nn.ParameterDict()
            for channel, idxs in self.mapper.items():
                self.weighted_sum_pooling[channel] = nn.Parameter(
                    torch.randn(len(idxs))
                )

        if self.pooling_channel_type in [
            ChannelPoolingType.WEIGHTED_SUM_ONE,
            ChannelPoolingType.WEIGHTED_SUM_ONE_NO_SOFTMAX,
        ]:
            self.weighted_sum_pooling = nn.Parameter(torch.ones(total_in_channels))

        if self.pooling_channel_type == ChannelPoolingType.ATTENTION:
            ## fill out some default values for attn_pooling_params
            attn_pooling_params.dim = self.kernels_per_channel
            attn_pooling_params.max_num_channels = total_in_channels
            self.attn_pooling = ChannelAttentionPoolingLayer(**attn_pooling_params)

        nn.init.kaiming_normal_(
            self.conv1depthwise_param_bank, mode="fan_in", nonlinearity="relu"
        )

        ## store reference for later access
        self.adaptive_interface = nn.ParameterList(
            [self.get_patch_emb, self.conv_1x1, self.conv1depthwise_param_bank]
        )
        if hasattr(self, "weighted_sum_pooling"):
            self.adaptive_interface.append(self.weighted_sum_pooling)
        if hasattr(self, "attn_pooling"):
            self.adaptive_interface.append(self.attn_pooling)

        ## shared feature_extractor
        self.feature_extractor = nn.Sequential(
            # model.stem[1],
            model.stages[0],
            model.stages[1],
            model.stages[2].downsample,
            *[model.stages[2].blocks[i] for i in range(9)],
            model.stages[3].downsample,
            *[model.stages[3].blocks[i] for i in range(3)],
        )

        self.norm = nn.InstanceNorm2d(out_dim, affine=True)

        num_proxies = (
            config.num_classes
        )  ## depends on the number of classes of the dataset
        self.dim = 768 if self.cfg.pooling in ["avg", "max", "avgmax"] else 7 * 7 * 768
        self.proxies = torch.nn.Parameter((torch.randn(num_proxies, self.dim) / 8))
        init_temperature = config.temperature  # scale = sqrt(1/T)
        if self.cfg.learnable_temp:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * np.log(1 / init_temperature)
            )
        else:
            self.scale = np.sqrt(1.0 / init_temperature)

    def slice_params_first_layer(self, chunk: str) -> Tensor:
        assert chunk in self.mapper, f"Invalid data_channel: {chunk}"

        ## conv1depthwise_param_bank's shape: (c_total * kernels_per_channel, 1, 3, 3)
        param_list = []
        for c in self.mapper[chunk]:
            param = self.conv1depthwise_param_bank[
                c * self.kernels_per_channel : (c + 1) * self.kernels_per_channel, ...
            ]
            param_list.append(param)
        params = torch.cat(param_list, dim=0)

        return params

    def _reset_params(self, model):
        for m in model.children():
            if len(list(m.children())) > 0:
                self._reset_params(m)

            elif isinstance(m, nn.Conv2d):
                print("resetting", m)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                print("resetting", m)

            elif isinstance(m, nn.Linear):
                print("resetting", m)

                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            else:
                print("skipped", m)

    def _init_bias(self, model):
        ## Init bias of the first layer
        if model.stem[0].bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.stem[0].weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(model.stem[0].bias, -bound, bound)

    def forward(self, x: torch.Tensor, chunk: str) -> torch.Tensor:
        c = x.shape[1]

        ## slice params of the first layers
        conv1depth_params = self.slice_params_first_layer(chunk)

        assert len(self.mapper[chunk]) == c
        assert conv1depth_params.shape == (c * self.kernels_per_channel, 1, 3, 3)

        out = self.get_patch_emb[chunk](x)
        out = F.conv2d(out, conv1depth_params, bias=None, stride=1, padding=1, groups=c)
        out = rearrange(out, "b (c k) h w -> b c k h w", k=self.kernels_per_channel)

        if self.pooling_channel_type == ChannelPoolingType.AVG:
            out = out.mean(dim=1)
        elif self.pooling_channel_type == ChannelPoolingType.SUM:
            out = out.sum(dim=1)
        elif self.pooling_channel_type in (
            ChannelPoolingType.WEIGHTED_SUM_RANDOM,
            ChannelPoolingType.WEIGHTED_SUM_ONE,
        ):
            weights = F.softmax(self.weighted_sum_pooling[self.mapper[chunk]])
            weights = rearrange(weights, "c -> c 1 1 1")
            out = (out * weights).sum(dim=1)
        elif self.pooling_channel_type in (
            ChannelPoolingType.WEIGHTED_SUM_RANDOM_NO_SOFTMAX,
            ChannelPoolingType.WEIGHTED_SUM_ONE_NO_SOFTMAX,
        ):
            weights = self.weighted_sum_pooling[self.mapper[chunk]]
            weights = rearrange(weights, "c -> c 1 1 1")
            out = (out * weights).sum(dim=1)
        elif (
            self.pooling_channel_type
            == ChannelPoolingType.WEIGHTED_SUM_RANDOM_PAIRWISE_NO_SOFTMAX
        ):
            weights = self.weighted_sum_pooling[chunk]
            weights = rearrange(weights, "c -> c 1 1 1")
            out = (out * weights).sum(dim=1)
        elif (
            self.pooling_channel_type == ChannelPoolingType.WEIGHTED_SUM_RANDOM_PAIRWISE
        ):
            weights = F.softmax(self.weighted_sum_pooling[chunk])
            weights = rearrange(weights, "c -> c 1 1 1")
            out = (out * weights).sum(dim=1)
        elif self.pooling_channel_type == ChannelPoolingType.ATTENTION:
            out = self.attn_pooling(out, channel_token_idxs=self.mapper[chunk])
        else:
            raise ValueError(
                f"Invalid pooling_channel_type: {self.pooling_channel_type}"
            )

        out = self.norm(self.conv_1x1(out))
        out = self.feature_extractor(out)
        if self.cfg.pooling == FeaturePooling.AVG:
            out = F.adaptive_avg_pool2d(out, (1, 1))
        elif self.cfg.pooling == FeaturePooling.MAX:
            out = F.adaptive_max_pool2d(out, (1, 1))
        elif self.cfg.pooling == FeaturePooling.AVG_MAX:
            x_avg = F.adaptive_avg_pool2d(out, (1, 1))
            x_max = F.adaptive_max_pool2d(out, (1, 1))
            out = torch.cat([x_avg, x_max], dim=1)
        elif self.cfg.pooling == FeaturePooling.NONE:
            pass
        else:
            raise ValueError(
                f"Pooling {self.cfg.pooling} not supported. Use one of {FeaturePooling.list()}"
            )
        out = rearrange(out, "b c h w -> b (c h w)")
        return out


def depthwiseconvnext(cfg: Model, **kwargs) -> DepthwiseConvNeXt:
    return DepthwiseConvNeXt(config=cfg, **kwargs)


if __name__ == "__main__":
    a = 1
    print(a)
