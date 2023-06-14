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


class ConvNeXtBase(nn.Module):
    def __init__(self, config: Model):
        # pretrained_model_name "convnext_tiny.fb_in22k"
        super().__init__()
        self.cfg = config
        model = create_model(config.pretrained_model_name, pretrained=config.pretrained)

        out_dim, original_in_dim, kh, kw = model.stem[0].weight.shape
        new_shape = (out_dim, config.in_dim, kh, kw)

        model = intialize_first_conv_layer(
            model,
            new_shape,
            original_in_dim,
            self.cfg.first_layer,
            self.cfg.in_dim,
            return_first_layer_only=False,
        )

        ## For logging purposes
        self.adaptive_interface = None

        ## Extractor: only use the feature extractor part of the model
        ## forward pass: https://github.com/huggingface/pytorch-image-models/blob/b3e816d6d71ec132b39c603d68b619ae2870fd0a/timm/models/convnext.py#L420
        self.feature_extractor = nn.Sequential(
            model.stem,
            model.stages[0],
            model.stages[1],
            model.stages[2].downsample,
            *[model.stages[2].blocks[i] for i in range(9)],
            model.stages[3].downsample,
            *[model.stages[3].blocks[i] for i in range(3)],
        )
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


def convnext_base(cfg: Model, **kwargs) -> ConvNeXtBase:
    return ConvNeXtBase(config=cfg)


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
    model = convnext_base(model_cfg)
    x = torch.randn(2, model_cfg.in_dim, 224, 224)
    y = model(x)
    print(y.shape)
