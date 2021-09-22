import pytest
import torch
from perceiver_pytorch.queries import LearnableQuery


@pytest.mark.parametrize("layer_shape", ["2d", "3d"])
def test_learnable_query(layer_shape):
    query_creator = LearnableQuery(
        channel_dim=32,
        query_shape=(24, 128, 128),
        conv_layer=layer_shape,
        max_frequency=64.0,
        frequency_base=2.0,
        num_frequency_bands=128,
        sine_only=False,
    )
    x = torch.randn((16, 24, 12, 128, 128))
    out = query_creator(x)

    pass
