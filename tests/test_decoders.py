from perceiver_pytorch.decoders import ImageDecoder
import pytest


@pytest.mark.parametrize("post_type", ["conv", "patches", "conv1x1", "pixels"])
def test_image_encoder(post_type):
    pass
