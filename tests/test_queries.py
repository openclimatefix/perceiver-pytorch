import pytest
import torch
from perceiver_pytorch.queries import LearnableQuery
from perceiver_pytorch.perceiver_io import PerceiverIO
import einops


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
    assert out.shape == (16, 393216, 803)


@pytest.mark.parametrize("layer_shape", ["2d", "3d"])
def test_learnable_query_qpplication(layer_shape):
    output_shape = (6, 32, 32)
    query_creator = LearnableQuery(
        channel_dim=32,
        query_shape=output_shape,
        conv_layer=layer_shape,
        max_frequency=64.0,
        frequency_base=2.0,
        num_frequency_bands=128,
        sine_only=False,
    )
    with torch.no_grad():
        query_creator.eval()
        x = torch.randn((4, 6, 12, 32, 32))
        out = query_creator(x)

        model = PerceiverIO(depth=2, dim=100, queries_dim=query_creator.output_shape()[-1])
        model.eval()
        model_input = torch.randn((4, 256, 100))
        model_out = model(model_input, queries=out)
        # Reshape back to correct shape
        model_out = einops.rearrange(
            model_out,
            "b (t h w) c -> b t c h w",
            t=output_shape[0],
            h=output_shape[1],
            w=output_shape[2],
        )
        assert model_out.shape == (4, 6, 803, 32, 32)
