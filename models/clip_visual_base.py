import numpy as np
import torch
from torch import nn
import clip

from config import Model
from helper_classes.first_layer_init import FirstLayerInit
from utils import convert_models_to_fp32


## try freeze BN
## try train proxies first, then fine tune

## Forward: https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py
class CLIPBasedModel(nn.Module):
    def __init__(self, config: Model):
        # "pretrained_model_name: RN50, ViT-B/16"
        super().__init__()
        self.cfg = config
        root_path = "/projectnb/ivc-ml/chaupham/pretrained_models/clip_openai"
        clip_model, preprocess = clip.load(config.pretrained_model_name, download_root=root_path,
                                           jit=False)  # jit=False for training
        model = clip_model.visual  ## only use the visual part of CLIP
        convert_models_to_fp32(model)
        self.snr = {}

        out_dim, original_in_dim, kh, kw = model.conv1.weight.shape
        new_shape = (out_dim, config.in_dim, kh, kw)

        ## Init weights and biases of the first layer
        if config.first_layer == FirstLayerInit.REINIT_AS_RANDOM:
            model.conv1.weight = nn.Parameter(torch.zeros(new_shape))
            nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
            # self._init_bias(model)
        elif config.first_layer == FirstLayerInit.PRETRAINED_PAD_RANDOM:
            if original_in_dim < config.in_dim:
                original_data = model.conv1.weight.data.detach().clone()
                model.conv1.weight = nn.Parameter(torch.zeros(new_shape))
                nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
                model.conv1.weight.data[:, :original_in_dim, :, :] = original_data
                # self._init_bias(model)
        elif config.first_layer == FirstLayerInit.PRETRAINED_PAD_AVG:
            if original_in_dim < config.in_dim:
                original_data = model.conv1.weight.data.detach().clone()
                model.conv1.weight = nn.Parameter(torch.zeros(new_shape))
                nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
                model.conv1.weight.data[:, :original_in_dim, :, :] = original_data

                num_channels_to_avg = 2 if config.in_dim == 5 else 3
                for i, c in enumerate(range(original_in_dim, config.in_dim)):
                    model.conv1.weight.data[:, c, :, :] = original_data[:, i:num_channels_to_avg + i, ...].mean(dim=1)
        else:
            raise NotImplementedError(f"First layer init {config.first_layer} not implemented")

        if config.pretrained_model_name == "RN50":
            ## Only use the feature extractor part of the model
            self.feature_extractor = nn.Sequential(*list(model.children()))
            ## freeze all the layers in feature extractor except some last layers
            num_unfrozen_layers = config.unfreeze_last_n_layers
            if num_unfrozen_layers > 0:
                for param in self.feature_extractor[:-num_unfrozen_layers].parameters():
                    param.requires_grad = False
            elif num_unfrozen_layers == 0:
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False

            if config.reset_last_n_unfrozen_layers and num_unfrozen_layers > 0:
                self.feature_extractor[-num_unfrozen_layers:].apply(self._reset_params)

            if config.unfreeze_first_layer:
                for p in self.feature_extractor[0].parameters():
                    p.requires_grad = True
        elif config.pretrained_model_name == "ViT-B/16":
            self.feature_extractor = model

            if not config.use_auto_rgn:
                ## freeze all model, except proj and conv1
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False

                ## pick some layers to fine-tune
                trainable_layers = config.unfreeze_vit_layers
                if "proj" in trainable_layers:
                    self.feature_extractor.proj.requires_grad = True
                if "conv1" in trainable_layers:
                    self.feature_extractor.conv1.weight.requires_grad = True
                if "pos_embed" in trainable_layers:
                    self.feature_extractor.pos_embed.requires_grad = True
                if "class_embed" in trainable_layers:
                    self.feature_extractor.class_embed.requires_grad = True
                if "ln_pre" in trainable_layers:
                    self.feature_extractor.ln_pre.weight.requires_grad = True
                    self.feature_extractor.ln_pre.bias.requires_grad = True
                if "ln_post" in trainable_layers:
                    self.feature_extractor.ln_post.weight.requires_grad = True
                    self.feature_extractor.ln_post.bias.requires_grad = True
                for i in range(12):
                    if f"block{i}" in trainable_layers:
                        for param in self.feature_extractor.transformer.resblocks[i].parameters():
                            param.requires_grad = True
            else:
                pass  # we train all layers, but each layer has its own learning rate, which is lr * rgn

        else:
            raise NotImplementedError(f"Pretrained model {config.pretrained_model_name} not implemented")

        num_proxies = config.num_classes  ## depends on the number of classes of the dataset
        self.dim = 1024 if config.pretrained_model_name == "RN50" else 512
        self.proxies = torch.nn.Parameter((torch.randn(num_proxies, self.dim) / 8))
        temperature = config.temperature  # scale = sqrt(1/T)
        self.scale = np.sqrt(1. / temperature)

    def _reset_params(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                print("resetting", m)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

    def _init_bias(self, model):
        ## Init bias of the first layer
        if model.conv1.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.conv1.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(model.conv1.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        return x


def clip_based_model(cfg: Model, **kwargs) -> CLIPBasedModel:
    model = CLIPBasedModel(config=cfg)
    # if cfg.use_auto_rgn:
    #     get_all_modules(model, assign_hook=True)   ## TODO: add Parameters (such as class_embedding, positional_embedding) in here as well?

    return model


def hook_fn_backward_wrapper(child_name):
    @torch.no_grad()
    def hook_fn_backward(module, grad_input, grad_output):
        """
        A hook function that will be called during backpropagation.
        """
        # Do something with the gradients, e.g. print or save to a file
        print(f"child_name: {child_name}")
        for name, weight in module.named_parameters():
            if weight.requires_grad:
                print("name", name)
                print("weight.shape", weight.shape)
                if grad_output is not None and grad_output[0] is not None:
                    print("grad_output.shape", grad_output[0].shape)
                    weight_norm = torch.norm(weight.data)
                    grad_norm = torch.norm(grad_output[0])
                    ratio1 = grad_norm / weight_norm
                    print("grad_norm", grad_norm.item())
                    print(f"ratio1 = {ratio1}")
                if grad_input is not None and grad_input[0] is not None:
                    print("grad_input.shape", grad_input[0].shape)
                    weight_norm = torch.norm(weight.data)
                    grad_norm2 = torch.norm(grad_input[0])
                    ratio2 = grad_norm2 / weight_norm
                    print("grad_norm2", grad_norm2.item())
                    print(f"ratio2= {ratio2}")
                print("-------")
                ## add results to snr
                # self.snr[f"snr/{name}"] = ratio
    return hook_fn_backward




def get_all_modules(network, assign_hook=False):
    def get_module_helper(network):
        for child_name, layer in network.named_children():
            # if leaf node, add it to list
            if list(layer.children()) == []:
                ## if it requires grad, add it to the list
                if any([p.requires_grad for p in layer.parameters()]):
                    all_layers.append(layer)
                    if assign_hook:
                        layer.register_full_backward_hook(hook_fn_backward_wrapper(child_name))
            else:
                get_module_helper(layer)

    all_layers = []
    get_module_helper(network)
    return all_layers




if __name__ == '__main__':
    model_cfg = Model(name="convnext_base", init_weights=True)
    model_cfg.pretrained_model_name = "RN50"
    model_cfg.pooling = "avg"
    model_cfg.unfreeze_last_n_layers = 1
    model_cfg.pretrained = False
    model_cfg.num_classes = 4
    model_cfg.temperature = 0.1
    model_cfg.first_layer = FirstLayerInit.PRETRAINED_PAD_AVG
    model_cfg.in_dim = 5
    model_cfg.reset_last_n_unfrozen_layers = True
    model = clip_based_model(model_cfg)
    x = torch.randn(2, model_cfg.in_dim, 224, 224)
    y = model(x)
    print(y.shape)
