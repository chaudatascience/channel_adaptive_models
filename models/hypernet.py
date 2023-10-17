import numpy as np
from einops import rearrange
from timm import create_model
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class HyperNetwork(nn.Module):
    def __init__(self, z_dim, d, kernel_size, out_size, in_size=1):
        super().__init__()
        self.z_dim = z_dim
        self.d = d  ## in the paper, d = z_dim
        self.kernel_size = kernel_size
        self.out_size = out_size
        self.in_size = in_size

        self.W = nn.Parameter(torch.randn((self.z_dim, self.in_size, self.d)))
        self.b = nn.Parameter(torch.randn((self.in_size, self.d)))

        self.W_out = nn.Parameter(
            torch.randn((self.d, self.out_size, self.kernel_size, self.kernel_size))
        )
        self.b_out = nn.Parameter(torch.randn((self.out_size, self.kernel_size, self.kernel_size)))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.W)
        nn.init.kaiming_normal_(self.W_out)

    def forward(self, z: Tensor) -> Tensor:
        """
        @param z: (num_channels, z_dim)
        @return: kernel (out_size, in_size, kernel_size, kernel_size)
        """
        a = torch.einsum("c z, z i d ->c i d", z, self.W) + self.b
        K = torch.einsum("c i d, d o h w ->c i o h w", a, self.W_out) + self.b_out
        K = rearrange(K, "c i o h w -> o (c i) h w")
        return K
