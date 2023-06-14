from copy import deepcopy

import numpy as np
from einops import rearrange
from timm import create_model
import torch
from torch import nn
import torch.nn.functional as F

from config import Model
from helper_classes.feature_pooling import FeaturePooling
from helper_classes.first_layer_init import FirstLayerInit
from models.model_utils import intialize_first_conv_layer


# model here: https://github.com/huggingface/pytorch-image-models/blob/b3e816d6d71ec132b39c603d68b619ae2870fd0a/timm/models/convnext.py#L410
# lr=4.0e-3 (mentioned in  A ConvNet for the 2020s paper)


class SharedConvNeXt(nn.Module):
    def __init__(self, config: Model):
        # pretrained_model_name "convnext_tiny.fb_in22k"
        ## forward pass: https://github.com/huggingface/pytorch-image-models/blob/b3e816d6d71ec132b39c603d68b619ae2870fd0a/timm/models/convnext.py#L420

        super().__init__()
        self.cfg = config
        model = create_model(config.pretrained_model_name, pretrained=config.pretrained)

        in_dim_map = {"Allen": 3, "HPA": 4, "CP": 5}

        self.first_layer = nn.ModuleDict()

        for chunk, new_in_dim in in_dim_map.items():
            layer_1 = self._get_first_layer(model, new_in_dim)
            self.first_layer.add_module(chunk, layer_1)

        ## Store reference to sel.first_layer for later access
        self.adaptive_interface = nn.ModuleList([self.first_layer])

        ## shared feature_extractor
        self.feature_extractor = nn.Sequential(
            model.stem[1],
            model.stages[0],
            model.stages[1],
            model.stages[2].downsample,
            *[model.stages[2].blocks[i] for i in range(9)],
            model.stages[3].downsample,
            *[model.stages[3].blocks[i] for i in range(3)],
        )

        ## Loss
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

    def _get_first_layer(self, model, new_in_dim):
        config = self.cfg
        config.in_dim = None

        out_dim, original_in_dim, kh, kw = model.stem[0].weight.shape
        new_shape = (out_dim, new_in_dim, kh, kw)
        layer_1 = model.stem[0].weight
        if config.first_layer == FirstLayerInit.REINIT_AS_RANDOM:
            layer_1 = nn.Parameter(torch.zeros(new_shape))
            nn.init.kaiming_normal_(layer_1, mode="fan_out", nonlinearity="relu")
            # self._init_bias(model)
        elif config.first_layer == FirstLayerInit.PRETRAINED_PAD_RANDOM:
            if original_in_dim < new_in_dim:
                original_data = model.stem[0].weight.data.detach().clone()
                layer_1 = nn.Parameter(torch.zeros(new_shape))
                nn.init.kaiming_normal_(layer_1, mode="fan_out", nonlinearity="relu")
                layer_1.data[:, :original_in_dim, :, :] = original_data
                # self._init_bias(model)
        elif config.first_layer == FirstLayerInit.PRETRAINED_PAD_AVG:
            if original_in_dim < new_in_dim:
                original_data = model.stem[0].weight.data.detach().clone()
                layer_1 = nn.Parameter(torch.zeros(new_shape))
                nn.init.kaiming_normal_(layer_1, mode="fan_out", nonlinearity="relu")
                layer_1.data[:, :original_in_dim, :, :] = original_data

                num_channels_to_avg = 2 if new_in_dim == 5 else 3
                for i, c in enumerate(range(original_in_dim, new_in_dim)):
                    layer_1.data[:, c, :, :] = original_data[
                        :, i : num_channels_to_avg + i, ...
                    ].mean(dim=1)
        else:
            raise NotImplementedError(
                f"First layer init {config.first_layer} not implemented"
            )
        conv1 = deepcopy(model.stem[0])
        conv1.weight = layer_1

        return conv1

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
        assert chunk in self.first_layer, f"chunk={chunk} is not valid!"
        conv1 = self.first_layer[chunk]
        x = conv1(x)
        x = self.feature_extractor(x)
        if self.cfg.pooling == FeaturePooling.AVG:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        elif self.cfg.pooling == FeaturePooling.MAX:
            x = F.adaptive_max_pool2d(x, (1, 1))
        elif self.cfg.pooling == FeaturePooling.AVG_MAX:
            x_avg = F.adaptive_avg_pool2d(x, (1, 1))
            x_max = F.adaptive_max_pool2d(x, (1, 1))
            x = torch.cat([x_avg, x_max], dim=1)
        elif self.cfg.pooling == FeaturePooling.NONE:
            pass
        else:
            raise ValueError(
                f"Pooling {self.cfg.pooling} not supported. Use one of {FeaturePooling.list()}"
            )
        x = rearrange(x, "b c h w -> b (c h w)")
        return x


def shared_convnext(cfg: Model, **kwargs) -> SharedConvNeXt:
    return SharedConvNeXt(config=cfg)


if __name__ == "__main__":
    model_cfg = Model(name="convnext_base", init_weights=True)
    model_cfg.pretrained_model_name = "convnext_tiny.fb_in22k"
    model_cfg.pooling = "avg"
    model_cfg.unfreeze_last_n_layers = 2
    model_cfg.pretrained = False
    model_cfg.num_classes = 4
    model_cfg.temperature = 0.1
    model_cfg.first_layer = FirstLayerInit.PRETRAINED_PAD_AVG
    model_cfg.in_dim = 4
    model_cfg.reset_last_n_unfrozen_layers = True
    model = SharedConvNeXt(model_cfg)
    x = torch.randn(2, model_cfg.in_dim, 224, 224)
    y = model(x, "hpa")
    print(y.shape)
