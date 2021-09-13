from math import log, pi

import torch
import torch.nn.functional as F
import numpy as np
import math
import einops


def extract_image_patches(
    x: torch.Tensor, kernel: int, stride: int = 1, dilation: int = 1
) -> torch.Tensor:
    """
    Extract image patches in a way similar to TensorFlow extract_image_patches
    Taken from https://discuss.pytorch.org/t/tf-extract-image-patches-in-pytorch/43837/8

    In the Perceiver JAX implementation they extract image patches matching TensorFlow's SAME padding.
    PyTorch doesn't have that same kind of option, so this is a way to do that.

    Args:
        x: Input Torch Tensor
        kernel: Size of kernel
        stride: Stride of patch
        dilation: Dilation rate

    Returns:
    Tensor of size [Batch, Height, Width, Channels*kernel*stride]

    """
    # Do TF 'SAME' Padding
    b, c, h, w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(
        x, (pad_row // 2, pad_row - pad_row // 2, pad_col // 2, pad_col - pad_col // 2)
    )

    # Extract patches
    # get all image windows of size (kernel, stride) and stride (kernel, stride)
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    # Permute so that channels are next to patch dimension
    patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()
    # View as [batch_size, height, width, channels*kh*kw]
    return patches.view(b, -1, patches.shape[-2], patches.shape[-1])


def reverse_space_to_depth(
    frames: torch.Tensor, temporal_block_size: int = 1, spatial_block_size: int = 1
) -> torch.Tensor:
    """Reverse space to depth transform.
    Works for images (dim = 4) and videos (dim = 5)"""
    if len(frames.shape) == 4:
        return einops.rearrange(
            frames,
            "b (dh dw c) h w -> b c (h dh) (w dw)",
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    elif len(frames.shape) == 5:
        return einops.rearrange(
            frames,
            "b t (dt dh dw c) h w -> b (t dt) c (h dh) (w dw)",
            dt=temporal_block_size,
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    else:
        raise ValueError(
            "Frames should be of rank 4 (batch, height, width, channels)"
            " or rank 5 (batch, time, height, width, channels)"
        )


def space_to_depth(
    frames: torch.Tensor, temporal_block_size: int = 1, spatial_block_size: int = 1
) -> torch.Tensor:
    """Space to depth transform.
    Works for images (dim = 4) and videos (dim = 5)"""
    if len(frames.shape) == 4:
        return einops.rearrange(
            frames,
            "b c (h dh) (w dw) -> b (dh dw c) h w",
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    elif len(frames.shape) == 5:
        return einops.rearrange(
            frames,
            "b (t dt) c (h dh) (w dw) -> b t (dt dh dw c) h w ",
            dt=temporal_block_size,
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    else:
        raise ValueError(
            "Frames should be of rank 4 (batch, height, width, channels)"
            " or rank 5 (batch, time, height, width, channels)"
        )


def encode_position(
    batch_size: int,
    axis: list,
    max_frequency: float,
    num_frequency_bands: int,
    frequency_base: float,
    sine_only: bool = False,
) -> torch.Tensor:
    """
    Encode the Fourier Features and return them

    Args:
        batch_size: Batch size
        axis: List containing the size of each axis
        max_frequency: Max frequency
        num_frequency_bands: Number of frequency bands to use
        frequency_base: Base frequency
        sine_only: (bool) Whether to only use Sine features or both Sine and Cosine, defaults to both

    Returns:
        Torch tensor containing the Fourier Features of shape [Batch, *axis]
    """
    axis_pos = list(
        map(
            lambda size: torch.linspace(-1.0, 1.0, steps=size),
            axis,
        )
    )
    pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
    enc_pos = fourier_encode(
        pos,
        max_frequency,
        num_frequency_bands,
        frequency_base,
        sine_only=sine_only,
    )
    enc_pos = einops.rearrange(enc_pos, "... n d -> ... (n d)")
    enc_pos = einops.repeat(enc_pos, "... -> b ...", b=batch_size)
    return enc_pos


def fourier_encode(
    x: torch.Tensor,
    max_freq: float,
    num_bands: int = 4,
    base: float = 2,
    sine_only: bool = False,
) -> torch.Tensor:
    """
    Create Fourier Encoding

    Args:
        x: Input Torch Tensor
        max_freq: Maximum frequency for the Fourier features
        num_bands: Number of frequency bands
        base: Base frequency
        sine_only: Whether to only use sine or both sine and cosine features

    Returns:
        Torch Tensor with the fourier position encoded concatenated
    """
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.logspace(
        0.0,
        log(max_freq / 2) / log(base),
        num_bands,
        base=base,
        device=device,
        dtype=dtype,
    )
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = x.sin() if sine_only else torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x
