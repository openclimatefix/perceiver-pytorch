from perceiver_pytorch.encoders import ImageEncoder
import torch
import pytest

@pytest.mark.parametrize("prep_type", ["conv", "patches", "conv1x1", "pixels"])
def test_image_encoder(prep_type):
    encoder = ImageEncoder(prep_type=prep_type, output_channels=48)
    image = torch.randn(2, 12, 256, 256)
    with torch.no_grad():
        out = encoder(image)
    print(out.shape)

