import torch
from einops import rearrange, repeat
from torch import nn

from perceiver_pytorch.layers import exists, cache_fn, PreNorm, FeedForward, Attention
from perceiver_pytorch.rotary import SinusoidalEmbeddings
from perceiver_pytorch.utils import encode_position

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
        weight_tie_layers=False,
        fourier_encode_data=True,
        sine_only: bool = False,
        self_per_cross_attn=1,
        self_attn_rel_pos=True,
    ):
        """
        Perceiver: https://arxiv.org/abs/2103.03206
        The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base of the logarithm function for Fourier position
              encoding.
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
            sine_only: Use only sine encoding in fourier encoding, compared to using sine and cos
          self_per_cross_attn: Number of self attention blocks per cross attn.
          self_attn_rel_pos:
        """
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.freq_base = freq_base

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        self.sine_only = sine_only
        input_dim = fourier_channels + input_channels

        # Randomly initialise the 'latent array'.
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        def get_cross_attn():
            return PreNorm(
                latent_dim,
                Attention(
                    latent_dim,
                    input_dim,
                    heads=cross_heads,
                    dim_head=cross_dim_head,
                    dropout=attn_dropout,
                ),
                context_dim=input_dim,
            )

        def get_cross_ff():
            return PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

        def get_latent_attn():
            return PreNorm(
                latent_dim,
                Attention(
                    latent_dim,
                    heads=latent_heads,
                    dim_head=latent_dim_head,
                    dropout=attn_dropout,
                ),
            )

        def get_latent_ff():
            return PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

        # Cache all the above functions.
        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(
            cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff)
        )

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {"_cache": should_cache}

            self_attns = nn.ModuleList([])

            for _ in range(self_per_cross_attn):
                self_attns.append(
                    nn.ModuleList(
                        [
                            get_latent_attn(**cache_args),
                            get_latent_ff(**cache_args),
                        ]
                    )
                )

            self.layers.append(
                nn.ModuleList(
                    [
                        get_cross_attn(**cache_args),
                        get_cross_ff(**cache_args),
                        self_attns,
                    ]
                )
            )

        self.to_logits = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, num_classes))

        self.sinu_emb = None
        if self_attn_rel_pos:
            self.sinu_emb = SinusoidalEmbeddings(latent_dim_head)

    def forward(self, data, mask=None):
        """
        Args:
          data: If sequential is True, then data must be of shape:
              (batch size, sequence length, *axes) where axes would be width
              and height for images.
        """

        b, *axis, _ = data.shape
        device = data.device

        assert (
            len(axis) == self.input_axis
        ), f"Input data must have {self.input_axis} axes, not {len(axis)}!"

        if self.fourier_encode_data:
            # Calculate Fourier encoded positions in the range of [-1, 1],
            # for all axes.
            enc_pos = encode_position(
                b,
                axis,
                self.max_freq,
                self.num_freq_bands,
                self.freq_base,
                sine_only=self.sine_only,
            ).type_as(data)

            data = torch.cat((data, enc_pos), dim=-1)

        # Concat to channels of data and flatten axes.
        # b = batch size; d = last dimension of data
        data = rearrange(data, "b ... d -> b (...) d", b=b)

        # x is the 'latent array' in the paper.
        # b = batch size; n = number of latents; d = latent dimensions.
        x = repeat(self.latents, "n d -> b n d", b=b)

        # Rotary embeddings for latents, if specified.
        pos_emb = self.sinu_emb(x) if exists(self.sinu_emb) else None

        # Layers.
        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x, pos_emb=pos_emb) + x
                x = self_ff(x) + x

        x = x.mean(dim=-2)
        return self.to_logits(x)
