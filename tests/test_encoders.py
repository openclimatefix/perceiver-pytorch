from perceiver_pytorch.encoders import ImageEncoder
import torch
import pytest


@pytest.mark.parametrize("prep_type", ["conv", "conv1x1"])
def test_conv_image_encoder(prep_type):
    encoder = ImageEncoder(prep_type=prep_type, output_channels=48)
    image = torch.randn(2, 12, 256, 256)
    with torch.no_grad():
        out = encoder(image)
    assert not torch.isnan(out).any(), "Output included NaNs"
    assert out.size() == (2, 48, 64, 64)


def test_patches_image_encoder():
    encoder = ImageEncoder(prep_type="patches", output_channels=48)
    image = torch.randn(2, 12, 256, 256)
    with torch.no_grad():
        out = encoder(image)
    assert not torch.isnan(out).any(), "Output included NaNs"
    assert out.size() == (2, 192, 64, 64)


def test_pixels_image_encoder():
    encoder = ImageEncoder(prep_type="pixels", output_channels=48)
    image = torch.randn(2, 12, 256, 256)
    with torch.no_grad():
        out = encoder(image)
    assert not torch.isnan(out).any(), "Output included NaNs"
    assert out.size() == (2, 12, 64, 64)


@pytest.mark.parametrize("prep_type", ["conv", "conv1x1"])
def test_conv_video_encoder(prep_type):
    encoder = ImageEncoder(prep_type=prep_type, output_channels=48)
    image = torch.randn(2, 6, 12, 256, 256)
    with torch.no_grad():
        out = encoder(image)
    assert not torch.isnan(out).any(), "Output included NaNs"
    assert out.size() == (2, 6, 48, 64, 64)


def test_patches_video_encoder():
    encoder = ImageEncoder(prep_type="patches", output_channels=48)
    image = torch.randn(2, 6, 12, 256, 256)
    with torch.no_grad():
        out = encoder(image)
    assert not torch.isnan(out).any(), "Output included NaNs"
    assert out.size() == (2, 6, 192, 64, 64)


def test_pixels_video_encoder():
    encoder = ImageEncoder(prep_type="pixels", output_channels=48)
    image = torch.randn(2, 6, 12, 256, 256)
    with torch.no_grad():
        out = encoder(image)
    assert not torch.isnan(out).any(), "Output included NaNs"
    assert out.size() == (2, 6, 12, 64, 64)


def test_pixels_video_downsample_encoder():
    encoder = ImageEncoder(
        prep_type="pixels", output_channels=48, temporal_downsample=2
    )
    image = torch.randn(2, 6, 12, 256, 256)
    with torch.no_grad():
        out = encoder(image)
    assert not torch.isnan(out).any(), "Output included NaNs"
    assert out.size() == (2, 3, 12, 64, 64)
