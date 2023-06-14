import numpy as np
from einops import rearrange
from timm import create_model
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from helper_classes.first_layer_init import FirstLayerInit

from config import Model
from helper_classes.feature_pooling import FeaturePooling
from models.model_utils import intialize_first_conv_layer


def get_module(module, name):
    for n, m in module.named_modules():
        if n == name:
            return m


def freeze_(model):
    """Freeze model. Note that this function does not control BN"""
    for p in model.parameters():
        p.requires_grad_(False)


class ConvNeXtSharedMIRO(nn.Module):
    """ConvNeXt + FrozenBN + IntermediateFeatures
    Based on https://github.com/kakaobrain/miro/blob/main/domainbed/networks/ur_networks.py#L165
    """

    def __init__(self, config: Model, freeze: str = None):
        # pretrained_model_name "convnext_tiny.fb_in22k"
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

        ## Extractor: only use the feature extractor part of the model
        ## forward pass: https://github.com/huggingface/pytorch-image-models/blob/b3e816d6d71ec132b39c603d68b619ae2870fd0a/timm/models/convnext.py#L420
        self.feature_extractor = nn.Sequential(
            model.stem[1],
            model.stages[0],
            model.stages[1],
            model.stages[2].downsample,
            *[model.stages[2].blocks[i] for i in range(9)],
            model.stages[3].downsample,
            *[model.stages[3].blocks[i] for i in range(3)],
        )
        self._features = []
        self.feat_layers = [
            "0",
            "1.blocks.2.drop_path",
            "2.blocks.2.drop_path",
            "12.drop_path",
            "16.drop_path",
        ]
        self.build_feature_hooks(self.feat_layers)

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

        self.freeze_bn()
        self.freeze(freeze)

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

    def hook(self, module, input, output):
        self._features.append(output)

    def build_feature_hooks(self, feat_layers):
        for n, m in self.feature_extractor.named_modules():
            if n in feat_layers:
                m.register_forward_hook(self.hook)
        return None

    def freeze_bn(self):
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze(self, freeze):
        if freeze is not None:
            if freeze == "all":
                print("Freezing all layers of the feature extractor")
                freeze_(self.feature_extractor)
            else:
                for block in self.blocks[: freeze + 1]:
                    freeze_(block)

    def clear_features(self):
        self._features.clear()

    def train(self, mode: bool = True):
        """Override the default train() to freeze the BN parameters"""
        super().train(mode)
        self.freeze_bn()

    def forward(
        self, x: torch.Tensor, chunk, return_features: bool = True
    ) -> torch.Tensor:
        self.clear_features()
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

        if return_features:
            return x, self._features
        else:
            return x


def convnext_shared_miro(cfg: Model, freeze: str = None) -> ConvNeXtSharedMIRO:
    return ConvNeXtSharedMIRO(config=cfg, freeze=freeze)
