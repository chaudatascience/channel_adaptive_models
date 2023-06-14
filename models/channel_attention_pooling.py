from typing import List

import torch
from torch import nn, Tensor

from einops import rearrange, repeat

from helper_classes.channel_initialization import ChannelInitialization


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, heads=8, dim_head=64, dropout=0.):
        """
        input x: (b, c, k)
        @param model_dim: k
        @param heads:
        @param dim_head:
        @param dropout:
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == model_dim)

        self.heads = heads
        self.scale = dim_head ** (-0.5)

        self.softmax = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(model_dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, model_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x: Tensor):
        """
        @param x: shape of  (b, c, k)
        @param channel_mask: shape of (c), to channel_mask if a channel is missing
        @return:
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b c (heads dim_head) -> b heads c dim_head', heads=self.heads), qkv)

        sim = torch.einsum('b h q d, b h k d -> b h q k', q, k) * self.scale
        # channel_mask out missing channels
        # sim = sim.masked_fill(channel_mask == torch.tensor(False), float("-inf"))
        attn = self.softmax(sim)

        out = torch.einsum('b h q k, b h k d -> b h q d', attn, v)
        out = rearrange(out, 'b h c d -> b c (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ChannelAttentionPoolingLayer(nn.Module):
    def __init__(self, max_num_channels: int, dim, depth, heads, dim_head, mlp_dim,
                 use_cls_token: bool,
                 use_channel_tokens: bool,
                 init_channel_tokens: ChannelInitialization, dropout=0.):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(dim)) if use_cls_token else None

        if use_channel_tokens:
            if init_channel_tokens == ChannelInitialization.RANDOM:
                self.channel_tokens = nn.Parameter(torch.randn(max_num_channels, dim)/8)
            elif init_channel_tokens == ChannelInitialization.ZERO:
                self.channel_tokens = nn.Parameter(torch.zeros(max_num_channels, dim))
            else:
                raise ValueError(f"init_channel_tokens {init_channel_tokens} not supported")
        else:
            self.channel_tokens = None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiHeadAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x: Tensor, channel_token_idxs: List[int]) -> Tensor:
        ## (batch, c, k, h, w) = > ((batch, h, w), c, k) = (b, c, k)
        batch, c, k, h, w = x.shape
        x = rearrange(x, 'b c k h w -> (b h w) c k')

        if self.channel_tokens is not None:
            tokens = self.channel_tokens[channel_token_idxs]
            x += tokens

        if self.cls_token is not None:
            cls_tokens = repeat(self.cls_token, 'k -> b 1 k', b=batch * h * w)
            x = torch.concat((cls_tokens, x), dim=1)

        for attention, feedforward in self.layers:
            x = attention(x) + x
            x = feedforward(x) + x
        x = rearrange(x, '(b h w) c k -> b c k h w', b=batch, h=h)

        if self.cls_token is not None:
            x = x[:, -1, ...]
        else:
            x = torch.mean(x, dim=1)

        return x


if __name__ == '__main__':
    batch, c, k, h, w = 2, 2, 6, 8, 10
    c_max = 3

    channels = [1, 2]
    # mask = torch.zeros(c_max, dtype=torch.bool)
    # mask[channels] = True
    # mask = rearrange(mask, 'c -> 1 1 1 c')

    x = torch.randn(batch, c, k, h, w)
    transformer = ChannelAttentionPoolingLayer(max_num_channels=c_max, dim=k, depth=1, heads=2, dim_head=k, mlp_dim=4,
                                               use_cls_token=True, use_channel_tokens=False, init_channel_tokens=None)
    y = transformer(x)
    print(y.shape)
