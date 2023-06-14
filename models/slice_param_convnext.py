import numpy as np
from einops import rearrange, repeat
from timm import create_model
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import repeat, rearrange

from config import Model
from helper_classes.feature_pooling import FeaturePooling
from helper_classes.first_layer_init import FirstLayerInit


# model here: https://github.com/huggingface/pytorch-image-models/blob/b3e816d6d71ec132b39c603d68b619ae2870fd0a/timm/models/convnext.py#L410
# lr=4.0e-3 (mentioned in  A ConvNet for the 2020s paper)


class SliceParamConvNeXt(nn.Module):
    def __init__(self, config: Model):
        # pretrained_model_name "convnext_tiny.fb_in22k"
        ## forward pass: https://github.com/huggingface/pytorch-image-models/blob/b3e816d6d71ec132b39c603d68b619ae2870fd0a/timm/models/convnext.py#L420

        super().__init__()
        self.cfg = config
        model = create_model(config.pretrained_model_name, pretrained=config.pretrained)

        self.duplicate = config.duplicate
        if self.duplicate:
            total_in_channels = 1
        else:
            total_in_channels = len(config.in_channel_names)

        ## all channels in this order (alphabet): ['er', 'golgi', 'membrane', 'microtubules','mito','nucleus','protein', 'rna']
        self.mapper = {
            "Allen": [5, 2, 6],
            "HPA": [3, 6, 5, 0],
            "CP": [5, 0, 7, 1, 4],
        }

        self.class_emb_idx = {
            "Allen": [0, 1, 2],
            "HPA": [3, 4, 5, 6],
            "CP": [7, 8, 9, 10, 11],
        }
        total_diff_class_channels = 12

        out_dim, original_in_dim, kh, kw = model.stem[0].weight.shape
        self.stride = model.stem[0].stride
        self.padding = model.stem[0].padding
        self.dilation = model.stem[0].dilation
        self.groups = model.stem[0].groups

        self.conv1_param_bank = nn.Parameter(
            torch.zeros(out_dim, total_in_channels, kh, kw)
        )
        self.init_slice_param_bank_(total_in_channels, model.stem[0].weight.data)

        if self.cfg.slice_class_emb:
            self.class_emb = nn.Parameter(
                torch.randn(out_dim, total_diff_class_channels, kh, kw) / 8
            )
        else:
            self.class_emb = None

        ## Make a list to store reference for easy access
        self.adaptive_interface = nn.ParameterList([self.conv1_param_bank])

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

    def init_slice_param_bank_(
        self, total_in_channels: int, conv1_weight: Tensor
    ) -> None:
        """
        Initialize the first layer of the model
        conv1_weight: pre-trained weight, shape (original_out_dim, original_in_dim, kh, kw)
        """
        if self.cfg.first_layer == FirstLayerInit.PRETRAINED_PAD_DUPS:
            ## copy all the weights from the pretrained model, duplicates the weights if needed'
            original_in_dim = conv1_weight.shape[1]
            num_dups = (total_in_channels // original_in_dim) + 1
            slice_params = repeat(conv1_weight, "o i h w -> o (i d) h w", d=num_dups)
            self.conv1_param_bank.data.copy_(slice_params[:, :total_in_channels])
        else:
            nn.init.kaiming_normal_(
                self.conv1_param_bank, mode="fan_in", nonlinearity="relu"
            )

    def slice_params_first_layer(self, chunk: str) -> Tensor:
        assert chunk in self.mapper, f"Invalid data_channel: {chunk}"
        if self.duplicate:
            ## conv1depthwise_param_bank's shape: (out_dim, 1, 3, 3)
            params = repeat(
                self.conv1_param_bank,
                "o i h w -> o (i c) h w",
                c=len(self.mapper[chunk]),
            )
        else:
            params = self.conv1_param_bank[:, self.mapper[chunk]]
            if self.class_emb is not None:
                params = params + self.class_emb[:, self.class_emb_idx[chunk]]
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
        conv1_params = self.slice_params_first_layer(chunk)
        x = F.conv2d(
            x,
            conv1_params,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

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


def sliceparamconvnext(cfg: Model, **kwargs) -> SliceParamConvNeXt:
    return SliceParamConvNeXt(config=cfg)


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
    model_cfg.in_channel_names = [
        "er",
        "golgi",
        "membrane",
        "microtubules",
        "mito",
        "nucleus",
        "protein",
        "rna",
    ]
    model_cfg.reset_last_n_unfrozen_layers = True
    model = SliceParamConvNeXt(model_cfg)
    x = torch.randn(96, model_cfg.in_dim, 224, 224)
    y = model(x, "hpa")
    print(y.shape)
