import numpy as np
import torch
from torch import nn

from config import Model


## DINO v2
# model here: https://github.com/facebookresearch/dinov2/blob/f8969297dbe53373b4041bf47d997a8dcc8d2077/dinov2/models/vision_transformer.py#L230

## Should we use patch tokens instead of cls token? or maybe both?


class DINOBase(nn.Module):
    def __init__(self, config: Model):
        super().__init__()
        self.cfg = config

        ## 'dinov2_vits14' 21M params
        model = torch.hub.load("facebookresearch/dinov2", config.pretrained_model_name)

        ## TODO: adapt with varying number of channels

        self.adaptive_interface = None

        ## TODO: add options to freeze some layers
        self.feature_extractor = model

        num_proxies = (
            config.num_classes
        )  ## depends on the number of classes of the dataset
        self.dim = (
            384 if self.cfg.pretrained_model_name == "dinov2_vits14" else 768
        )  ## Base: 768
        self.proxies = torch.nn.Parameter((torch.randn(num_proxies, self.dim) / 8))
        init_temperature = config.temperature  # scale = sqrt(1/T)
        if self.cfg.learnable_temp:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * np.log(1 / init_temperature)
            )
        else:
            self.scale = np.sqrt(1.0 / init_temperature)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        return x


def dino_base(cfg: Model, **kwargs) -> DINOBase:
    return DINOBase(config=cfg)
