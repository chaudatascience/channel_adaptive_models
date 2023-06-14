import numpy as np
from einops import rearrange
from timm import create_model
import torch
from torch import nn
import torch.nn.functional as F

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


class ConvNeXtBaseMIRO(nn.Module):
    """ConvNeXt + FrozenBN + IntermediateFeatures
    Based on https://github.com/kakaobrain/miro/blob/main/domainbed/networks/ur_networks.py#L165
    """

    def __init__(self, config: Model, freeze: str = None):
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
        self._features = []
        self.feat_layers = [
            "0.1",
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

    def forward(self, x: torch.Tensor, return_features: bool = True) -> torch.Tensor:
        self.clear_features()

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


def convnext_base_miro(cfg: Model, freeze: str = None) -> ConvNeXtBaseMIRO:
    return ConvNeXtBaseMIRO(config=cfg, freeze=freeze)


## test model
if __name__ == "__main__":
    from config import Model
    import debugpy

    # debugpy.listen(5678)
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    in_dim = 5
    config = Model(
        pretrained_model_name="convnext_tiny.fb_in22k",
        in_dim=in_dim,
        first_layer="pretrained_pad_avg",
        num_classes=10,
        pooling="avg",
        temperature=1.0,
        learnable_temp=False,
        pretrained=False,
        name="convnext_tiny",
        init_weights=True,
    )
    model = ConvNeXtBaseMIRO(config, freeze="all")
    res = model(torch.randn(1, in_dim, 224, 224), return_features=True)[1]
    for x in res:
        print(x.shape)

    """
    torch.Size([1, 96, 56, 56])
    torch.Size([1, 96, 56, 56])
    torch.Size([1, 192, 28, 28])
    torch.Size([1, 384, 14, 14])
    torch.Size([1, 768, 7, 7])
    """
