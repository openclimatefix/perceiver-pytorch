import torch
import numpy as np
import math

from perceiver_pytorch.convolutions import Conv2DDownsample
from perceiver_pytorch.utils import space_to_depth


class ImageEncoder(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 12,
        prep_type: str = "conv",
        spatial_downsample: int = 4,
        temporal_downsample: int = 1,
        output_channels: int = 64,
        conv_after_patching: bool = False,
        conv2d_use_batchnorm: bool = True,
    ):
        """
        Image encoder class, modeled off the JAX version
        https://github.com/deepmind/deepmind-research/blob/769bfdbeafbcb472cb8e2c6cfa746b53ac82efc2/perceiver/io_processors.py#L291-L438

        Args:
            input_channels: Number of input channels of the original image/video
            prep_type: How to encode the images, one of conv, patches, pixels, or conv1x1
            spatial_downsample: How much to downsample spatially
            temporal_downsample: How much to downsample temporally
            output_channels: Number of output channels to send to Perceiver
            conv_after_patching: Whether to use convolutions after creating patches
            conv2d_use_batchnorm: Whether to use batch norm
        """
        super().__init__()
        self.conv_after_patching = conv_after_patching
        self.prep_type = prep_type
        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample
        self.output_channels = output_channels

        if prep_type not in ("conv", "patches", "pixels", "conv1x1"):
            raise ValueError("Invalid prep_type!")

        if self.prep_type == "conv":
            # Downsampling with conv is currently restricted
            convnet_num_layers = math.log(spatial_downsample, 4)
            convnet_num_layers_is_int = convnet_num_layers == np.round(
                convnet_num_layers
            )
            if not convnet_num_layers_is_int or temporal_downsample != 1:
                raise ValueError(
                    "Only powers of 4 expected for spatial "
                    "and 1 expected for temporal "
                    "downsampling with conv."
                )

            self.convnet = Conv2DDownsample(
                num_layers=int(convnet_num_layers),
                output_channels=output_channels,
                input_channels=input_channels,
                use_batchnorm=conv2d_use_batchnorm,
            )
        elif self.prep_type == "conv1x1":
            assert temporal_downsample == 1, "conv1x1 does not downsample in time."
            self.convnet_1x1 = torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(1, 1),
                # spatial_downsample is unconstrained for 1x1 convolutions.
                stride=(spatial_downsample, spatial_downsample),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prep_type == "conv":
            if len(x.shape) == 5:
                # Timeseries, do it to each timestep independently
                outs = []
                for i in range(x.shape[1]):
                    outs.append(self.convnet(x[:, i, :, :, :]))
                x = torch.stack(outs, dim=1)
            else:
                x = self.convnet(x)

        elif self.prep_type == "conv1x1":
            if len(x.shape) == 5:
                # Timeseries, do it to each timestep independently
                outs = []
                for i in range(x.shape[1]):
                    outs.append(self.convnet_1x1(x[:, i, :, :, :]))
                x = torch.stack(outs, dim=1)
            else:
                x = self.convnet_1x1(x)

        elif self.prep_type == "patches":

            x = space_to_depth(
                x,
                temporal_block_size=self.temporal_downsample,
                spatial_block_size=self.spatial_downsample,
            )

            # For flow
            if x.ndim == 5 and x.shape[1] == 1:
                x = x.squeeze(axis=1)
        elif self.prep_type == "pixels":
            # If requested, will downsample in simplest way
            if x.ndim == 4:
                x = x[:, :, :: self.spatial_downsample, :: self.spatial_downsample]
            elif x.ndim == 5:
                x = x[
                    :,
                    :: self.temporal_downsample,
                    :,
                    :: self.spatial_downsample,
                    :: self.spatial_downsample,
                ]
            else:
                raise ValueError("Unsupported data format for pixels")

        return x
