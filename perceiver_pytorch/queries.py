import torch
from torch.distributions import uniform
from typing import List, Union, Tuple
from math import prod


class LearnableQuery(torch.nn.Module):
    """
    Module that constructs a learnable query of query_shape for the Perceiver
    """
    def __init__(self, query_dim: int, query_shape: Union[Tuple[int], List[int]]):
        """
        Learnable Query with some inbuilt randomness to help with ensembling

        Args:
            query_dim: Query dimension
            query_shape: The final shape of the query, generally, the (T, H, W) of the output
        """
        super().__init__()
        self.query_shape = prod(query_shape) # Flatten the shape
        self.query_dim = query_dim
        self.learnable_query = torch.nn.Linear(self.query_dim, self.query_dim)
        self.distribution = uniform.Uniform(low=torch.Tensor([0.0]), high=torch.Tensor([1.0]))

    def forward(self, x: torch.Tensor):
        z = self.distribution.sample(
            (x.shape[0], self.query_future_size, self.query_dim)
        ).type_as(x)
        queries = self.learnable_query(z)
        return queries
