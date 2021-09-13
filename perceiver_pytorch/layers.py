from functools import wraps

import torch
from einops import rearrange, repeat
from torch import nn, einsum
from torch.nn import functional as F

from perceiver_pytorch.rotary import apply_rotary_emb


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = (
            nn.LayerNorm(context_dim) if exists(context_dim) else None
        )

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    """
    Gaussian Error Gated Linear Unit.
    See Shazer 2020: https://arxiv.org/abs/2002.05202
    """
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    """Feed forward neural net with GEGLU activation."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        """
        Args:
            dim: Input & Output size.
            mult: The inner dimension of the FF net will be dim * mult.
            dropout: Proportion to dropout after the GEGLU.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0
    ):
        """
        Args:
            query_dim: Size of the queries.
            context_dim: Size of the 'context' (the 'byte array' in the paper).
                If None, will default to the query_dim.
            heads: Number of attention heads.
            dim_head: Number of dimensions per head.
            dropout: Proportion to dropout (in the final linear layer).
        """

        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, pos_emb=None):
        """

        Args:
            x: The 'latent array' in the Perceiver paper.
            context: The 'byte array' in the Perceiver paper (the input data).
            mask:
            pos_emb:

        Returns:

        """

        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # Rearrange the query, key and value tensors.
        # b = batch size; n = TODO (PD-2021-09-13)
        # h = number of heads; d = number of dims per head.
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v)
        )

        if exists(pos_emb):
            q, k = apply_rotary_emb(q, k, pos_emb)

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)
