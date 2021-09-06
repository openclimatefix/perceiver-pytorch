import torch

from perceiver_pytorch.decoders import ImageDecoder
import pytest


def test_conv_image_decoder():
    decoder = ImageDecoder(postprocess_type="conv", output_channels=12, input_channels=48, spatial_upsample=4)
    inputs = torch.randn(2, 48, 64, 64)
    with torch.no_grad():
        out = decoder(inputs)
    assert not torch.isnan(out).any(), "Output included NaNs"
    assert out.size() == (2, 12, 256, 256)


def test_conv1x1_image_decoder():
    decoder = ImageDecoder(postprocess_type="conv1x1", output_channels=12, input_channels=48, spatial_upsample=4)
    inputs = torch.randn(2, 48, 64, 64)
    with torch.no_grad():
        out = decoder(inputs)
    assert not torch.isnan(out).any(), "Output included NaNs"
    # Conv1x1 downsample if spatial_upsample > 1
    assert out.size() == (2, 12, 16, 16)


def test_patches_image_decoder():
    decoder = ImageDecoder(postprocess_type="patches", spatial_upsample=4)
    inputs = torch.randn(2, 192, 64, 64)
    with torch.no_grad():
        out = decoder(inputs)
    assert not torch.isnan(out).any(), "Output included NaNs"
    assert out.size() == (2, 12, 256, 256)


def test_pixel_image_decoder():
    decoder = ImageDecoder(postprocess_type="pixels")
    inputs = torch.randn(2, 192, 64, 64)
    with torch.no_grad():
        out = decoder(inputs)
    assert not torch.isnan(out).any(), "Output included NaNs"
    assert out.size() == (2, 192, 64, 64)
    assert pytest.approx(inputs, out)


def test_conv_video_decoder():
    decoder = ImageDecoder(postprocess_type="conv", output_channels=12, input_channels=48, spatial_upsample=4)
    inputs = torch.randn(2, 3, 48, 64, 64)
    with torch.no_grad():
        out = decoder(inputs)
    assert not torch.isnan(out).any(), "Output included NaNs"
    assert out.size() == (2, 3, 12, 256, 256)


def test_conv1x1_video_decoder():
    decoder = ImageDecoder(postprocess_type="conv1x1", output_channels=12, input_channels=48, spatial_upsample=4)
    inputs = torch.randn(2, 3, 48, 64, 64)
    with torch.no_grad():
        out = decoder(inputs)
    assert not torch.isnan(out).any(), "Output included NaNs"
    # Conv1x1 downsample if spatial_upsample > 1
    assert out.size() == (2, 3, 12, 16, 16)


@pytest.mark.skip("This test does not work because the Conv3DUpsample decoder is off by a few timesteps Jacob Sept 2021 - see issue #7")
def test_conv3d_video_decoder():
    decoder = ImageDecoder(postprocess_type="conv", output_channels=12, input_channels=48, spatial_upsample=4, temporal_upsample=2)
    inputs = torch.randn(2, 1, 48, 64, 64)
    with torch.no_grad():
        out = decoder(inputs)
    assert not torch.isnan(out).any(), "Output included NaNs"
    assert out.size() == (2, 2, 12, 256, 256)


def test_patches_video_decoder():
    decoder = ImageDecoder(postprocess_type="patches", spatial_upsample=4)
    inputs = torch.randn(2, 3, 192, 64, 64)
    with torch.no_grad():
        out = decoder(inputs)
    assert not torch.isnan(out).any(), "Output included NaNs"
    assert out.size() == (2, 3, 12, 256, 256)


def test_pixel_video_decoder():
    decoder = ImageDecoder(postprocess_type="pixels")
    inputs = torch.randn(2, 3, 192, 64, 64)
    with torch.no_grad():
        out = decoder(inputs)
    assert not torch.isnan(out).any(), "Output included NaNs"
    assert out.size() == (2, 3, 192, 64, 64)
    assert pytest.approx(inputs, out)
