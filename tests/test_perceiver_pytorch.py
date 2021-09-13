from perceiver_pytorch.perceiver_pytorch import Perceiver
import torch


def test_init_model():

    _ = Perceiver(
        input_channels=16,
        input_axis=2,
        num_freq_bands=6,
        max_freq=10,
        depth=13,
        num_latents=16,
        latent_dim=17,
        num_classes=7,
        weight_tie_layers=True,
        fourier_encode_data=False,
    )


def test_model_forward():

    model = Perceiver(
        input_channels=16,
        input_axis=2,
        num_freq_bands=6,
        max_freq=10,
        depth=13,
        num_latents=16,
        latent_dim=17,
        num_classes=7,
        weight_tie_layers=True,
        fourier_encode_data=False,
    )

    x = torch.randn(8 * 13, 32, 32, 16)
    y = model(x)


def test_model_forward_fourier():

    model = Perceiver(
        input_channels=16,
        input_axis=2,
        num_freq_bands=6,
        max_freq=10,
        depth=13,
        num_latents=16,
        latent_dim=17,
        num_classes=7,
        weight_tie_layers=True,
        fourier_encode_data=True,
    )

    x = torch.randn(8 * 13, 32, 32, 16)
    y = model(x)
