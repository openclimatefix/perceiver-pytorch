import torch
from torch.distributions import uniform
from typing import List, Union, Tuple
from math import prod
from perceiver_pytorch.utils import encode_position


class LearnableQuery(torch.nn.Module):
    """
    Module that constructs a learnable query of query_shape for the Perceiver
    """

    def __init__(
        self,
        query_dim: int,
        query_shape: Union[Tuple[int], List[int]],
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
        self.query_shape = prod(query_shape)  # Flatten the shape
        # Need to get Fourier Features once and then just append to the output
        self.fourier_features = encode_position(
            1,
            axis=query_shape,
            max_frequency=max_frequency,
            frequency_base=frequency_base,
            num_frequency_bands=num_frequency_bands,
            sine_only=sine_only,
        )
        self.query_dim = query_dim
        self.learnable_query = torch.nn.Linear(self.query_dim, self.query_dim)
        self.distribution = uniform.Uniform(low=torch.Tensor([0.0]), high=torch.Tensor([1.0]))

    def forward(self, x: torch.Tensor):
        z = self.distribution.sample((x.shape[0], self.query_future_size, self.query_dim)).type_as(
            x
        )
        # Do 3D or 2D CNN to keep same spatial size, concat, then linearize
        queries = self.learnable_query(z)
        # TODO Add Fourier Features
        return queries
