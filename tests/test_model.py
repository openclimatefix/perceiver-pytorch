import torch
from einops import rearrange
from perceiver_pytorch.multi_perceiver_pytorch import MultiPerceiver
from perceiver_pytorch.modalities import InputModality


def test_multiperceiver_creation():
    # Timeseries input
    input_size = 64
    max_frequency = 16.0
    video_modality = InputModality(
        name="timeseries",
        input_channels=12,
        input_axis=3,  # number of axes, 3 for video
        num_freq_bands=input_size,  # number of freq bands, with original value (2 * K + 1)
        max_freq=max_frequency,  # maximum frequency, hyperparameter depending on how fine the data is, should be Nyquist frequency (i.e. 112 for 224 input image)
        sin_only=False,  # Whether if sine only for Fourier encoding, TODO test more
        fourier_encode=True,  # Whether to encode position with Fourier features
    )
    # Use image modality for latlon, elevation, other base data?
    image_modality = InputModality(
        name="base",
        input_channels=4,
        input_axis=2,  # number of axes, 2 for images
        num_freq_bands=input_size,  # number of freq bands, with original value (2 * K + 1)
        max_freq=max_frequency,  # maximum frequency, hyperparameter depending on how fine the data is
        sin_only=False,
        fourier_encode=True,
    )
    # Sort audio for timestep one-hot encode? Or include under other modality?
    timestep_modality = InputModality(
        name="forecast_time",
        input_channels=1,  # number of channels for mono audio
        input_axis=1,  # number of axes, 2 for images
        num_freq_bands=24,  # number of freq bands, with original value (2 * K + 1)
        max_freq=16.0,  # maximum frequency, hyperparameter depending on how fine the data is
        sin_only=False,
        fourier_encode=True,
    )
    model = MultiPerceiver(
        modalities=[video_modality, image_modality, timestep_modality],
        queries_dim=input_size,
        depth=6,
        forecast_steps=12,
        output_shape=input_size,
    )
    x = {
        "timeseries": torch.randn((2, 6, input_size, input_size, 12)),
        "base": torch.randn((2, input_size, input_size, 4)),
        "forecast_time": torch.randn(2, 24, 1),
    }
    query = torch.randn((2, input_size * 12, input_size))
    model.eval()
    with torch.no_grad():
        out = model(x, queries=query)
        out = rearrange(
            out, "b h (w c) -> b c h w", c=12
        )
    # MetNet creates predictions for the center 1/4th
    assert out.size() == (
        2,
        12,
        12 * input_size,
        input_size,
    )
    assert not torch.isnan(out).any(), "Output included NaNs"
