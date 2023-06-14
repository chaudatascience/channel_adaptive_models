## https://github.com/microsoft/Swin-Transformer/blob/f92123a0035930d89cf53fcb8257199481c4428d/optimizer.py
from torch import nn
from timm.optim import AdamP, AdamW
import torch

from utils import read_yaml


def make_my_optimizer(opt_name: str, model_params, cfg: dict):
    opt_name = opt_name.lower()
    if opt_name == 'sgd':
        optimizer = torch.optim.SGD(model_params, **cfg)
    elif opt_name == 'adam':
        # https://stackoverflow.com/questions/64621585/adamw-and-adam-with-weight-decay
        # https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html
        optimizer = torch.optim.Adam(model_params, **cfg)
    elif opt_name == 'adamw':
        optimizer = AdamW(model_params, **cfg)
    elif opt_name == 'adamp':
        optimizer = AdamP(model_params, **cfg)
    else:
        raise NotImplementedError(f'Not implemented optimizer: {opt_name}')

    return optimizer


if __name__ == '__main__':
    conf = read_yaml('configs/cifar/optimizer/adamw.yaml')
    model = nn.Linear(3, 4)

    optimizer = make_my_optimizer('adamw', model.parameters(), conf['params'])
    print(optimizer.state_dict())
