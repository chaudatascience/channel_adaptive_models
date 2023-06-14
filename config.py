from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional

from omegaconf import MISSING

from helper_classes.channel_initialization import ChannelInitialization
from helper_classes.feature_pooling import FeaturePooling
from helper_classes.first_layer_init import FirstLayerInit
from helper_classes.norm_type import NormType
from helper_classes.channel_pooling_type import ChannelPoolingType

# fmt: off

@dataclass
class OptimizerParams(Dict):
    pass


@dataclass
class Optimizer:
    name: str
    params: OptimizerParams


@dataclass
class SchedulerParams(Dict):
    pass


@dataclass
class Scheduler:
    name: str
    convert_to_batch: bool
    params: SchedulerParams


@dataclass
class Train:
    batch_strategy: None
    resume_train: bool
    resume_model: str
    use_amp: bool
    checkpoints: str
    clip_grad_norm: int
    batch_size: int
    num_epochs: int
    verbose_batches: int
    seed: int
    save_model: bool
    debug: Optional[bool] = False
    real_batch_size: Optional[int] = None
    compile_pytorch: Optional[bool] = False
    adaptive_interface_epochs: int = 0
    adaptive_interface_lr: Optional[float] = None
    swa: Optional[bool] = False
    swad: Optional[bool] = False
    swa_lr: Optional[float] = 0.05
    swa_start: Optional[int] = 5

    ## MIRO
    miro: Optional[bool] = False
    miro_lr_mult: Optional[float] = 10.0
    miro_ld: Optional[float] = 0.01  # 0.1

@dataclass
class Eval:
    batch_size: int
    dest_dir: str = ""  ## where to save results
    feature_dir: str = ""  ## where to save features for evaluation
    root_dir: str = ""  ## folder that contains images and metadata
    classifiers: List[str] = field(default_factory=list)  ## classifier to use
    classifier: str = ""  ## placeholder for classifier
    feature_file: str = ""  ## feature file to use
    use_gpu: bool = True  ## use gpu for evaluation
    knn_metrics: List[str] = field(default_factory=list)  ## "l2" or "cosine"
    knn_metric: str = ""  ## should be "l2" or "cosine", placeholder
    meta_csv_file: str = ""  ## metadata csv file
    clean_up: bool = True  ## whether to delete the feature file after evaluation
    only_eval_first_and_last: bool = False  ## whether to only evaluate first (off the shelf) and last (final fune-tuned) epochs

@dataclass
class AttentionPoolingParams:
    """
    param for ChannelAttentionPoolingLayer class.
    initialize all arguments in the class.
    """

    max_num_channels: int
    dim: int
    depth: int
    dim_head: int
    heads: int
    mlp_dim: int
    dropout: float
    use_cls_token: bool
    use_channel_tokens: bool
    init_channel_tokens: ChannelInitialization


@dataclass
class Model:
    name: str
    init_weights: bool
    in_dim: int = MISSING
    num_classes: int = MISSING  ## Num of training classes
    freeze_other: Optional[bool] = None  ## used in Shared Models
    in_channel_names: Optional[List[str]] = None  ## used with Slice Param Models
    separate_norm: Optional[
        bool
    ] = None  ## use a separate norm layer for each data chunk
    image_h_w: Optional[List[int]] = None  ## used with layer norm
    norm_type: Optional[
        NormType
    ] = None  # one of ["batch_norm", "norm_type", "instance_norm"]
    duplicate: Optional[
        bool
    ] = None  # whether to only use the first param bank and duplicate for all the channels
    pooling_channel_type: Optional[ChannelPoolingType] = None
    kernels_per_channel: Optional[int] = None
    num_templates: Optional[int] = None  # number of templates to use in template mixing
    separate_coef: Optional[bool] = None  # whether to use a separate set of coefficients for each chunk
    coefs_init: Optional[bool] = None # whether to initialize the coefficients, used in templ mixing ver2
    freeze_coefs_epochs: Optional[int] = None # TODO: add this. Whether to freeze the coefficients for some first epoch, used in templ mixing ver2
    separate_emb: Optional[bool] = None  # whether to use a separate embedding (hypernetwork) for each chunk
    z_dim: Optional[int] = None  # dimension of the latent space, hypernetwork
    hidden_dim: Optional[int] = None  # dimension of the hidden layer, hypernetwork

    ### ConvNet/CLIP-ResNet50 Params
    pretrained: Optional[bool] = None
    pretrained_model_name: Optional[str] = None
    pooling: Optional[FeaturePooling] = None  # one of ["avg", "max", "avgmax", "none"]
    temperature: Optional[float] = None
    unfreeze_last_n_layers: Optional[int] = -1
    # -1: unfreeze all layers, 0: freeze all layers, 1: unfreeze last layer, etc.
    first_layer: Optional[FirstLayerInit] = None
    unfreeze_first_layer: Optional[bool] = True
    reset_last_n_unfrozen_layers: Optional[bool] = False
    use_auto_rgn: Optional[bool] = None  # relative gradient norm, this supersedes the use of `unfreeze_vit_layers`

    ### CLIP ViT16Base
    unfreeze_vit_layers: Optional[List[str]] = None

    ## temperature in the loss
    learnable_temp: bool = False

    ## Slice Params
    slice_class_emb: Optional[bool] = False


@dataclass
class Dataset:
    name: str
    img_size: int = 224
    label_column: Optional[str] = None
    root_dir: str = ""
    file_name: str = ""


@dataclass
class Wandb:
    use_wandb: bool
    log_freq: int
    num_images_to_log: int
    log_imgs_every_n_epochs: int
    project_name: str


@dataclass
class Logging:
    wandb: Wandb
    use_py_log: bool
    scc_jobid: Optional[str] = None


@dataclass
class DataChunk:
    chunks: List[Dict[str, List[str]]]

    def __str__(self) -> str:
        channel_names = [list(c.keys())[0] for c in self.chunks]
        channel_values = [list(c.values())[0] for c in self.chunks]

        channels = zip(*(channel_names, channel_values))
        channels_str = "----".join(
            ["--".join([c[0], "_".join(c[1])]) for c in channels]
        )
        return channels_str


@dataclass
class Hardware:
    num_workers: int
    device: str
    multi_gpus: str


@dataclass
class MyConfig:
    train: Train
    eval: Eval
    optimizer: Optimizer
    scheduler: Scheduler
    model: Model
    dataset: Dataset
    data_chunk: DataChunk
    logging: Logging
    hardware: Hardware
    tag: str
    attn_pooling: Optional[AttentionPoolingParams] = None

