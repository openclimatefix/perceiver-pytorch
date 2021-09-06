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
    frames: np.ndarray, temporal_block_size: int = 1, spatial_block_size: int = 1
) -> np.ndarray:
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
    frames: np.ndarray, temporal_block_size: int = 1, spatial_block_size: int = 1
) -> np.ndarray:
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
