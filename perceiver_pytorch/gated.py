import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

from perceiver_pytorch.layers import exists, default, cache_fn, PreNorm, FeedForward, Attention
from perceiver_pytorch.utils import fourier_encode


# helpers


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class GRUGating(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.dim = dim
        self.fn = fn
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, x, **kwargs):
        b, dim = x.shape[0], self.dim
        y = self.fn(x, **kwargs)

        gated_output = self.gru(
            rearrange(y, "... d -> (...) d"), rearrange(x, "... d -> (...) d")
        )

        gated_output = rearrange(gated_output, "(b n) d -> b n d", b=b)
        return gated_output


# main class


class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        freq_base=2,
        input_channels=3,
        input_axis=2,
        num_latents=512,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        num_classes=1000,
        attn_dropout=0.0,
        ff_dropout=0.0,
        weight_tie_layers=False
    ):
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.freq_base = freq_base

        input_dim = input_axis * ((num_freq_bands * 2) + 1) + input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: GRUGating(
            latent_dim,
            PreNorm(
                latent_dim,
                Attention(
                    latent_dim,
                    input_dim,
                    heads=cross_heads,
                    dim_head=cross_dim_head,
                    dropout=attn_dropout,
                ),
                context_dim=input_dim,
            ),
        )
        get_latent_attn = lambda: GRUGating(
            latent_dim,
            PreNorm(
                latent_dim,
                Attention(
                    latent_dim,
                    heads=latent_heads,
                    dim_head=latent_dim_head,
                    dropout=attn_dropout,
                ),
            ),
        )
        get_cross_ff = lambda: Residual(
            PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
        )
        get_latent_ff = lambda: Residual(
            PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
        )

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(
            cache_fn,
            (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff),
        )

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {"_cache": should_cache}

            self.layers.append(
                nn.ModuleList(
                    [
                        get_cross_attn(**cache_args),
                        get_cross_ff(**cache_args),
                        get_latent_attn(**cache_args),
                        get_latent_ff(**cache_args),
                    ]
                )
            )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, num_classes)
        )

    def forward(self, data, mask=None):
        b, *axis, _, device = *data.shape, data.device
        assert (
            len(axis) == self.input_axis
        ), "input data must have the right number of axis"

        # calculate fourier encoded positions in the range of [-1, 1], for all axis

        axis_pos = list(
            map(
                lambda size: torch.linspace(
                    -1.0, 1.0, steps=size, device=device
                ),
                axis,
            )
        )
        pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
        enc_pos = fourier_encode(
            pos, self.max_freq, self.num_freq_bands, base=self.freq_base
        )
        enc_pos = rearrange(enc_pos, "... n d -> ... (n d)")
        enc_pos = repeat(enc_pos, "... -> b ...", b=b)

        # concat to channels of data and flatten axis

        data = torch.cat((data, enc_pos), dim=-1)
        data = rearrange(data, "b ... d -> b (...) d")

        x = repeat(self.latents, "n d -> b n d", b=b)

        for cross_attn, cross_ff, latent_attn, latent_ff in self.layers:
            x = cross_attn(x, context=data, mask=mask)
            x = cross_ff(x)
            x = latent_attn(x)
            x = latent_ff(x)

        x = x.mean(dim=-2)
        return self.to_logits(x)
