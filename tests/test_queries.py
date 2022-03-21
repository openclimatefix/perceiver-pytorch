import einops
import pytest
import torch

from perceiver_pytorch.perceiver_io import PerceiverIO
from perceiver_pytorch.queries import LearnableQuery
from perceiver_pytorch.utils import encode_position


@pytest.mark.parametrize("layer_shape", ["2d", "3d"])
def test_learnable_query(layer_shape):
    query_creator = LearnableQuery(
        channel_dim=32,
        query_shape=(6, 16, 16),
        conv_layer=layer_shape,
        max_frequency=64.0,
        num_frequency_bands=128,
        sine_only=False,
        generate_fourier_features=True,
    )
    x = torch.randn((4, 6, 12, 16, 16))
    out = query_creator(x)
    # Output is flattened, so should be [B, T*H*W, C]
    # Channels is from channel_dim + 3*(num_frequency_bands * 2 + 1)
    # 32 + 3*(257) = 771 + 32 = 803
    assert out.shape == (4, 16 * 16 * 6, 803)


@pytest.mark.parametrize("layer_shape", ["2d", "3d"])
def test_learnable_query_no_fourier(layer_shape):
    query_creator = LearnableQuery(
        channel_dim=32,
        query_shape=(6, 16, 16),
        conv_layer=layer_shape,
        max_frequency=64.0,
        num_frequency_bands=128,
        sine_only=False,
        generate_fourier_features=False,
    )
    x = torch.randn((4, 6, 12, 16, 16))
    out = query_creator(x)
    assert out.shape == (4, 16 * 16 * 6, 32)


@pytest.mark.parametrize("layer_shape", ["2d", "3d"])
def test_learnable_query_qpplication(layer_shape):
    output_shape = (6, 16, 16)
    query_creator = LearnableQuery(
        channel_dim=32,
        query_shape=output_shape,
        conv_layer=layer_shape,
        max_frequency=64.0,
        num_frequency_bands=32,
        sine_only=False,
        generate_fourier_features=True,
    )
    with torch.no_grad():
        query_creator.eval()
        x = torch.randn((2, 6, 12, 16, 16))
        out = query_creator(x)

        model = PerceiverIO(depth=2, dim=100, queries_dim=query_creator.output_shape()[-1])
        model.eval()
        model_input = torch.randn((2, 256, 100))
        model_out = model(model_input, queries=out)
        # Reshape back to correct shape
        model_out = einops.rearrange(
            model_out,
            "b (t h w) c -> b t c h w",
            t=output_shape[0],
            h=output_shape[1],
            w=output_shape[2],
        )
        assert model_out.shape == (2, 6, 227, 16, 16)


@pytest.mark.parametrize("layer_shape", ["2d", "3d"])
def test_learnable_query_precomputed_fourier_only(layer_shape):
    precomputed_features = encode_position(
        1,  # Batch size, 1 for this as it will be adapted in forward
        axis=(10, 16, 16),  # 4 history + 6 future steps
        max_frequency=16.0,
        num_frequency_bands=128,
        sine_only=False,
    )
    # Only take future ones
    precomputed_features = precomputed_features[:, 4:]
    query_creator = LearnableQuery(
        channel_dim=32,
        query_shape=(6, 16, 16),
        conv_layer=layer_shape,
        max_frequency=64.0,
        num_frequency_bands=16,
        sine_only=False,
        precomputed_fourier=precomputed_features,
        generate_fourier_features=False,
    )
    x = torch.randn((4, 6, 12, 16, 16))
    out = query_creator(x)
    # Output is flattened, so should be [B, T*H*W, C]
    # Channels is from channel_dim + 3*(num_frequency_bands * 2 + 1)
    # 32 + 3*(257) = 771 + 32 = 803
    assert out.shape == (4, 16 * 16 * 6, 803)


@pytest.mark.parametrize("layer_shape", ["2d", "3d"])
def test_learnable_query_precomputed_and_generated_fourer(layer_shape):
    precomputed_features = encode_position(
        1,  # Batch size, 1 for this as it will be adapted in forward
        axis=(10, 16, 16),  # 4 history + 6 future steps
        max_frequency=16.0,
        num_frequency_bands=128,
        sine_only=False,
    )
    # Only take future ones
    precomputed_features = precomputed_features[:, 4:]
    query_creator = LearnableQuery(
        channel_dim=32,
        query_shape=(6, 16, 16),
        conv_layer=layer_shape,
        max_frequency=64.0,
        num_frequency_bands=128,
        sine_only=False,
        precomputed_fourier=precomputed_features,
        generate_fourier_features=True,
    )
    x = torch.randn((4, 6, 12, 16, 16))
    out = query_creator(x)
    # Output is flattened, so should be [B, T*H*W, C]
    # Channels is from channel_dim + 3*(num_frequency_bands * 2 + 1)
    # 32 + 3*(257) = 771 + 32 = 803
    # Then add 771 from the precomputed features, to get 803 + 771
    assert out.shape == (4, 16 * 16 * 6, 803 + 771)


@pytest.mark.parametrize("layer_shape", ["2d", "3d"])
def test_learnable_query_pass_in_fourier(layer_shape):
    precomputed_features = encode_position(
        4,
        axis=(10, 16, 16),  # 4 history + 6 future steps
        max_frequency=16.0,
        num_frequency_bands=64,
        sine_only=False,
    )
    # Only take future ones
    precomputed_features = precomputed_features[:, 4:]
    query_creator = LearnableQuery(
        channel_dim=32,
        query_shape=(6, 16, 16),
        conv_layer=layer_shape,
        max_frequency=64.0,
        num_frequency_bands=128,
        sine_only=False,
        generate_fourier_features=False,
    )
    x = torch.randn((4, 6, 12, 16, 16))
    out = query_creator(x, precomputed_features)
    # Output is flattened, so should be [B, T*H*W, C]
    # Channels is from channel_dim + 3*(num_frequency_bands * 2 + 1)
    # 3*(129) = 389 + 32 = 419
    # Since this is less than what is passed to LearnableQuery, we know its using the passed in features
    assert out.shape == (4, 16 * 16 * 6, 419)


@pytest.mark.parametrize("layer_shape", ["2d", "3d"])
def test_learnable_query_all_fouriers(layer_shape):
    batch_ff = encode_position(
        4,
        axis=(10, 16, 16),  # 4 history + 6 future steps
        max_frequency=16.0,
        num_frequency_bands=32,
        sine_only=False,
    )
    # Only take future ones
    batch_ff = batch_ff[:, 4:]
    precomputed_features = encode_position(
        1,
        axis=(10, 16, 16),  # 4 history + 6 future steps
        max_frequency=16.0,
        num_frequency_bands=64,
        sine_only=False,
    )
    # Only take future ones
    precomputed_features = precomputed_features[:, 4:]
    query_creator = LearnableQuery(
        channel_dim=32,
        query_shape=(6, 16, 16),
        conv_layer=layer_shape,
        max_frequency=64.0,
        num_frequency_bands=128,
        sine_only=False,
        precomputed_fourier=precomputed_features,
        generate_fourier_features=True,
    )
    x = torch.randn((4, 6, 12, 16, 16))
    out = query_creator(x, batch_ff)
    # Output is flattened, so should be [B, T*H*W, C]
    # Channels is from channel_dim + 3*(num_frequency_bands * 2 + 1)
    # 3*(129) = 389 + 32 = 419 + 771 from the generated ones + 195 from the batch features
    # Since this is less than what is passed to LearnableQuery, we know its using the passed in features
    assert out.shape == (4, 16 * 16 * 6, 1385)
