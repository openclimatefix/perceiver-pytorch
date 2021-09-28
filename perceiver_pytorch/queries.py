import torch
from torch.distributions import uniform
from typing import List, Union, Tuple
from perceiver_pytorch.utils import encode_position
from math import prod
import einops
import logging

_LOG = logging.getLogger("perceiver.queries")
_LOG.setLevel(logging.WARN)


class LearnableQuery(torch.nn.Module):
    """
    Module that constructs a learnable query of query_shape for the Perceiver
    """

    def __init__(
        self,
        channel_dim: int,
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
            channel_dim: Channel dimension for the output of the network
            query_shape: The final shape of the query, generally, the (T, H, W) of the output
        """
        super(LearnableQuery, self).__init__()
        self.query_shape = query_shape
        # Need to get Fourier Features once and then just append to the output
        self.fourier_features = encode_position(
            1,  # Batch size, 1 for this as it will be adapted in forward
            axis=query_shape,
            max_frequency=max_frequency,
            frequency_base=frequency_base,
            num_frequency_bands=num_frequency_bands,
            sine_only=sine_only,
        )
        self.channel_dim = channel_dim
        if (
            conv_layer == "3d" and len(self.query_shape) == 3
        ):  # If Query shape is for an image, then 3D conv won't work
            conv = torch.nn.Conv3d
        elif conv_layer == "2d":
            conv = torch.nn.Conv2d
        else:
            raise ValueError(f"Value for 'layer' is {conv_layer} which is not one of '3d', '2d'")
        self.conv_layer = conv_layer
        self.layer = conv(
            in_channels=channel_dim, out_channels=channel_dim, kernel_size=3, padding=1
        )
        # Linear layer to compress channels down to query_dim size?
        self.fc = torch.nn.Linear(self.channel_dim, self.channel_dim)
        self.distribution = uniform.Uniform(low=torch.Tensor([0.0]), high=torch.Tensor([1.0]))

    def output_shape(self) -> Tuple[int, int]:
        """
        Gives the output shape from the query, useful for setting the correct
        query_dim in the Perceiver

        Returns:
            The shape of the resulting query, excluding the batch size
        """

        # The shape is the query_dim + Fourier Feature channels
        channels = self.fourier_features.shape[-1] + self.channel_dim
        return prod(self.query_shape), channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Samples the uniform distribution and creates the query by passing the
        sample through the model and appending Fourier features

        Args:
            x: The input tensor to the model, used to batch the batch size

        Returns:
            Torch tensor used to query the output of the PerceiverIO model
        """
        _LOG.debug(f"Batch: {x.shape[0]} Query: {self.query_shape} Dim: {self.channel_dim}")
        z = self.distribution.sample((x.shape[0], self.channel_dim, *self.query_shape)).type_as(
            x
        )  # [B, Query, T, H, W, 1] or [B, Query, H, W, 1]
        z = torch.squeeze(z, dim=-1)  # Extra 1 for some reason
        _LOG.debug(f"Z: {z.shape}")
        # Do 3D or 2D CNN to keep same spatial size, concat, then linearize
        if self.conv_layer == "2d" and len(self.query_shape) == 3:
            # Iterate through time dimension
            outs = []
            for i in range(x.shape[1]):
                outs.append(self.layer(z[:, :, i, :, :]))
            query = torch.stack(outs, dim=2)
        else:
            query = self.layer(z)
        # Add Fourier Features
        ff = einops.repeat(
            self.fourier_features, "b ... -> (repeat b) ...", repeat=x.shape[0]
        )  # Match batches
        # Move channels to correct location
        query = einops.rearrange(query, "b c ... -> b ... c")
        query = torch.cat([query, ff], dim=-1)
        # concat to channels of data and flatten axis
        query = einops.rearrange(query, "b ... d -> b (...) d")
        _LOG.debug(f"Final Query Shape: {query.shape}")
        return query
