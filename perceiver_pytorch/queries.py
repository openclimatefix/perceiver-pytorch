import torch
from torch.distributions import uniform
from typing import List, Union, Tuple
from math import prod
from perceiver_pytorch.utils import encode_position
import einops


class LearnableQuery(torch.nn.Module):
    """
    Module that constructs a learnable query of query_shape for the Perceiver
    """

    def __init__(
        self,
        query_dim: int,
        query_shape: Union[Tuple[int], List[int]],
        conv_layer: str = "3d",
        max_frequency: float = 16.0,
        num_frequency_bands: int = 64,
        frequency_base: float = 2.0,
        sine_only: bool = False,
    ):
        """
        Learnable Query with some inbuilt randomness to help with ensembling

        Args:
            query_dim: Query dimension
            query_shape: The final shape of the query, generally, the (T, H, W) of the output
        """
        super().__init__()
        self.query_shape = query_shape
        # Need to get Fourier Features once and then just append to the output
        self.fourier_features = encode_position(
            1,
            axis=query_shape,
            max_frequency=max_frequency,
            frequency_base=frequency_base,
            num_frequency_bands=num_frequency_bands,
            sine_only=sine_only,
        )
        print(self.fourier_features.shape)
        self.query_dim = query_dim
        if conv_layer == "3d":
            conv = torch.nn.Conv3d
        elif conv_layer == "2d":
            conv = torch.nn.Conv2d
        else:
            raise ValueError(f"Value for 'layer' is {layer} which is not one of '3d', '2d'")
        self.conv_layer = conv_layer
        self.layer = conv(in_channels=query_dim, out_channels=query_dim, kernel_size=3, padding=1)
        # Linear layer to compress channels down to query_dim size?
        self.fc = torch.nn.Linear(self.query_dim, self.query_dim)
        self.distribution = uniform.Uniform(low=torch.Tensor([0.0]), high=torch.Tensor([1.0]))

    def forward(self, x: torch.Tensor):
        print(f"Batch: {x.shape[0]} Query: {self.query_shape} Dim: {self.query_dim}")
        z = self.distribution.sample((x.shape[0], self.query_dim, *self.query_shape)).type_as(
            x
        )  # [B, Query, T, H, W, 1] or [B, Query, H, W, 1]
        z = torch.squeeze(z, dim=-1)  # Extra 1 for some reason
        print(f"Z: {z.shape}")
        # Do 3D or 2D CNN to keep same spatial size, concat, then linearize
        if self.conv_layer == "2d":
            # Iterate through time dimension
            outs = []
            for i in range(x.shape[1]):
                outs.append(self.layer(z[:, i, :, :, :]))
            query = torch.stack(outs, dim=1)
        else:
            query = self.layer(z)
        # Add Fourier Features
        ff = einops.repeat(
            self.fourier_features, "b ... -> (repeat b) ...", repeat=x.shape[0]
        )  # Match batches
        print(ff.shape)
        query = torch.cat([query, ff], dim=-1)
        print(query.shape)
        # concat to channels of data and flatten axis
        query = einops.rearrange(query, "b ... d -> b (...) d")
        # Need query to end with query_dim channels

        return query
