from torch import nn, Tensor
import torch.nn.functional as F

from utils import pairwise_distance_v2


def proxy_loss(proxies, img_emb, gt_imgs, scale: float | nn.Parameter) -> Tensor:
    ## https://arxiv.org/pdf/2004.01113v2.pdf
    proxies_emb = scale * F.normalize(proxies, p=2, dim=-1)
    img_emb = scale * F.normalize(img_emb, p=2, dim=-1)

    img_dist = pairwise_distance_v2(proxies=proxies_emb, x=img_emb, squared=True)
    img_dist = img_dist * -1.0

    cross_entropy = nn.CrossEntropyLoss(reduction="mean")
    img_loss = cross_entropy(img_dist, gt_imgs)
    return img_loss
