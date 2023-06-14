from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .wrn import wrn28_10
from .slice_param_resnet import sliceparamresnet18
from .depthwise_resnet import depthwiseresnet18
from .template_mixing_first_layer_resnet import templatemixingfirstlayerresnet18
from .hypernetwork_resnet import hypernetwork_resnet18
from .convnext_base import convnext_base
from .convnext_base_miro import convnext_base_miro
from .convnext_shared_miro import convnext_shared_miro
from .clip_visual_base import clip_based_model
from .shared_convnext import shared_convnext
from .slice_param_convnext import sliceparamconvnext
from .template_mixing_convnext import templatemixingconvnext
from .template_convnextv2 import templatemixingconvnextv2
from .hypernet_convnext import hyperconvnext
from .hypernet_convnext_miro import hyperconvnext_miro
from .depthwise_convnext import depthwiseconvnext
from .depthwise_convnext_miro import depthwiseconvnext_miro
from .slice_param_convnext_miro import sliceparamconvnext_miro
from .dino_base import dino_base
from .template_convnextv2_miro import templatemixingconvnextv2_miro

## for new model, add the new model in _forward_model(), trainer.py
