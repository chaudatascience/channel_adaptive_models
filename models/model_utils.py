from typing import Tuple
import torch
from torch import nn
from itertools import chain
from timm.models import ConvNeXt
import torch.nn.functional as F
from config import Model
from helper_classes.first_layer_init import FirstLayerInit


def conv1x1(in_dim: int, out_dim: int, stride: int = 1) -> nn.Conv2d:
    """return 1x1 conv"""
    return nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_dim: int, out_dim: int, stride: int, padding: int) -> nn.Conv2d:
    """return 3x3 conv"""
    return nn.Conv2d(
        in_dim, out_dim, kernel_size=3, stride=stride, padding=padding, bias=False
    )


def toggle_grad(model: nn.Module, requires_grad: bool) -> None:
    """Toggle requires_grad for all parameters in the model"""
    for param in model.parameters():
        param.requires_grad = requires_grad
    return None


def unfreeze_last_layers(model: nn.Module, num_last_layers: int) -> None:
    """Freeze some last layers of the feature extractor"""
    if num_last_layers == -1:  #### unfreeze all
        toggle_grad(model, requires_grad=True)
    else:
        ## First, freeze all
        toggle_grad(model, requires_grad=False)

        ## Then, unfreeze some last layers
        if num_last_layers > 0:
            for param in model[-num_last_layers:].parameters():
                param.requires_grad = True
        else:  ## == 0
            pass  ## all layers are frozen
    return None


def intialize_first_conv_layer(
    model: ConvNeXt,
    new_shape: Tuple,
    original_in_dim: int,
    first_layer: FirstLayerInit,
    in_dim: int,
    return_first_layer_only: bool,
):
    """
    Initialize the first conv layer of a ConvNext model
    Return a modified model with the first conv layer initialized
    """
    ###### Init weights and biases of the first layer
    ## Note: we can also use the built-in feature of timm for this by using `in_chans`
    ## model = create_model(config.pretrained_model_name, pretrained=config.pretrained,  in_chans=config.in_dim)
    ## Read more: https://timm.fast.ai/models#So-how-is-timm-able-to-load-these-weights
    if first_layer == FirstLayerInit.REINIT_AS_RANDOM:
        model.stem[0].weight = nn.Parameter(torch.zeros(new_shape))
        nn.init.kaiming_normal_(
            model.stem[0].weight, mode="fan_out", nonlinearity="relu"
        )
        # self._init_bias(model)
    elif first_layer == FirstLayerInit.PRETRAINED_PAD_RANDOM:
        if original_in_dim < in_dim:
            original_data = model.stem[0].weight.data.detach().clone()
            model.stem[0].weight = nn.Parameter(torch.zeros(new_shape))
            nn.init.kaiming_normal_(
                model.stem[0].weight, mode="fan_out", nonlinearity="relu"
            )
            model.stem[0].weight.data[:, :original_in_dim, :, :] = original_data
            model.stem[0].in_channels = in_dim

            # self._init_bias(model)
    elif first_layer == FirstLayerInit.PRETRAINED_PAD_AVG:
        if original_in_dim < in_dim:
            original_data = model.stem[0].weight.data.detach().clone()
            model.stem[0].weight = nn.Parameter(torch.zeros(new_shape))
            nn.init.kaiming_normal_(
                model.stem[0].weight, mode="fan_out", nonlinearity="relu"
            )
            model.stem[0].weight.data[:, :original_in_dim, :, :] = original_data

            num_channels_to_avg = (
                2 if in_dim == 5 else 3
            )  ## TODO: make this more generic
            for i, c in enumerate(range(original_in_dim, in_dim)):
                model.stem[0].weight.data[:, c, :, :] = original_data[
                    :, i : num_channels_to_avg + i, ...
                ].mean(dim=1)
            model.stem[0].in_channels = in_dim
    else:
        raise NotImplementedError(f"First layer init {first_layer} not implemented")
    if return_first_layer_only:
        return model.stem[0]
    return model


class MeanEncoder(nn.Module):
    """Identity function"""

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x


class VarianceEncoder(nn.Module):
    """Bias-only model with diagonal covariance"""

    def __init__(self, shape, init=0.1, channelwise=True, eps=1e-5):
        super().__init__()
        self.shape = shape
        self.eps = eps

        init = (torch.as_tensor(init - eps).exp() - 1.0).log()
        b_shape = shape
        if channelwise:
            if len(shape) == 4:
                # [B, C, H, W]
                b_shape = (1, shape[1], 1, 1)
            elif len(shape) == 3:
                # CLIP-ViT: [H*W+1, B, C]
                b_shape = (1, 1, shape[2])
            else:
                raise ValueError()

        self.b = nn.Parameter(torch.full(b_shape, init))

    def forward(self, x):
        return F.softplus(self.b) + self.eps


def get_shapes(model, input_shape):
    # get shape of intermediate features
    with torch.no_grad():
        dummy = torch.rand(1, *input_shape).to(next(model.parameters()).device)
        try:
            _, feats = model(dummy)
        except:
            _, feats = model(dummy, chunk="Allen")
        shapes = [f.shape for f in feats]

    return shapes


def zip_strict(*iterables):
    """strict version of zip. The length of iterables should be same.

    NOTE yield looks non-reachable, but they are required.
    """
    # For trivial cases, use pure zip.
    if len(iterables) < 2:
        return zip(*iterables)

    # Tail for the first iterable
    first_stopped = False

    def first_tail():
        nonlocal first_stopped
        first_stopped = True
        return
        yield

    # Tail for the zip
    def zip_tail():
        if not first_stopped:
            raise ValueError("zip_equal: first iterable is longer")
        for _ in chain.from_iterable(rest):
            raise ValueError("zip_equal: first iterable is shorter")
            yield

    # Put the pieces together
    iterables = iter(iterables)
    first = chain(next(iterables), first_tail())
    rest = list(map(iter, iterables))
    return chain(zip(first, *rest), zip_tail())


def get_module(module, name):
    for n, m in module.named_modules():
        if n == name:
            return m


def freeze_(model):
    """Freeze model. Note that this function does not control BN"""
    for p in model.parameters():
        p.requires_grad_(False)
