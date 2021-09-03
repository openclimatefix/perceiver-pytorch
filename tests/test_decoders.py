import torch

from perceiver_pytorch.decoders import ImageDecoder
import pytest


@pytest.mark.parametrize("post_type", ["conv", "patches", "conv1x1", "pixels"])
def test_image_decoder(post_type):
    decoder = ImageDecoder(postprocess_type=post_type, output_channels=12, input_channels=48, spatial_upsample=4)
    inputs = torch.randn(2, 192, 64, 64)
    with torch.no_grad():
        out = decoder(inputs)
    assert not torch.isnan(out).any(), "Output included NaNs"
    assert out.size() == (2, 12, 256, 256)
