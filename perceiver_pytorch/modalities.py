import torch
from dataclasses import dataclass


@dataclass
class InputModality:
    name: str
    input_channels: int
    input_axis: int
    num_freq_bands: int
    max_freq: float
    freq_base: int = 2
    sin_only: bool = False
    fourier_encode: bool = True

    @property
    def input_dim(self) -> int:
        # Calculate the dimension of this modality.
        fourier_channels = self.input_axis * ((self.num_freq_bands * 2) + 1)
        fourier_channels = fourier_channels // 2 if self.sin_only else fourier_channels
        input_dim = fourier_channels + self.input_channels
        return input_dim


def modality_encoding(
    batch_size: int, axes, modality_index: int, num_modalities: int
) -> torch.Tensor:
    """
    Return one-hot encoding of modality given num_modalities, batch size and axes.
    The result need to be compatible with the modality data for concatenation.

    Args:
        batch_size: Batch size of the input
        axes: The size of each axis, other than batch size, of the input
        modality_index:  The index of this modality i.e. if there are 3 modalities, this would be 0, 1, or 2
        num_modalities: Total number of modalities

    Returns:
        One hot encoding of which modality the input is

    """
    one_hot = torch.eye(num_modalities, num_modalities)[modality_index]
    to_expand = [batch_size]
    one_hot = one_hot.unsqueeze(0)
    for i, axis in enumerate(axes):
        one_hot = one_hot.unsqueeze(0)
        to_expand.append(axis)
    to_expand.append(num_modalities)

    one_hot = one_hot.expand(to_expand)
    return one_hot
