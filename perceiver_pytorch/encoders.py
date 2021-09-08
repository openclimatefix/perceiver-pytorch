import torch
from torch import nn
import torchvision
import torch.nn.functional as F
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
        conv2d_use_batchnorm: bool = True,
        crop_size: int = 256,
        use_space2depth: bool = True,
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
            conv2d_use_batchnorm: Whether to use batch norm
            crop_size: Only for MetNet preprocessor, the center crop size
            use_space2depth: Only for MetNet preprocessor, whether to use average pooling, or space2depth for downsampling
        """
        super().__init__()
        self.prep_type = prep_type

        if prep_type not in ("conv", "patches", "pixels", "conv1x1", "metnet"):
            raise ValueError("Invalid prep_type!")

        if self.prep_type == "conv":
            self.encoder = ImageEncoderConv(
                input_channels=input_channels,
                temporal_downsample=temporal_downsample,
                spatial_downsample=spatial_downsample,
                output_channels=output_channels,
                conv2d_use_batchnorm=conv2d_use_batchnorm,
            )
        elif self.prep_type == "conv1x1":
            self.encoder = ImageEncoderConv1x1(
                input_channels=input_channels,
                spatial_downsample=spatial_downsample,
                output_channels=output_channels,
            )
        elif self.prep_type == "patches":
            self.encoder = ImageEncoderPatches(
                temporal_downsample=temporal_downsample,
                spatial_downsample=spatial_downsample,
            )
        elif self.prep_type == "pixels":
            self.encoder = ImageEncoderPixel(
                temporal_downsample=temporal_downsample,
                spatial_downsample=spatial_downsample,
            )
        elif self.prep_type == "metnet":
            self.encoder = ImageEncoderMetNet(
                crop_size=crop_size, use_space2depth=use_space2depth
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ImageEncoderConv(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 12,
        spatial_downsample: int = 4,
        temporal_downsample: int = 1,
        output_channels: int = 64,
        conv2d_use_batchnorm: bool = True,
    ):
        """
        Convolutional image encoder that can spatially and temporally downsample

        Args:
            input_channels: Number of input channels of the original image/video
            spatial_downsample: How much to downsample spatially
            temporal_downsample: How much to downsample temporally
            output_channels: Number of output channels to send to Perceiver
            conv2d_use_batchnorm: Whether to use batch norm
        """
        super().__init__()
        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample
        self.output_channels = output_channels

        # Downsampling with conv is currently restricted
        convnet_num_layers = math.log(spatial_downsample, 4)
        convnet_num_layers_is_int = convnet_num_layers == np.round(convnet_num_layers)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 5:
            # Timeseries, do it to each timestep independently
            outs = []
            for i in range(x.shape[1]):
                outs.append(self.convnet(x[:, i, :, :, :]))
            x = torch.stack(outs, dim=1)
        else:
            x = self.convnet(x)
        return x


class ImageEncoderConv1x1(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 12,
        spatial_downsample: int = 4,
        output_channels: int = 64,
    ):
        """
        Convolutional 1x1 encoder that can spatially downsample

        Args:
            input_channels: Number of input channels of the original image/video
            spatial_downsample: How much to downsample spatially
            output_channels: Number of output channels to send to Perceiver
        """
        super().__init__()
        self.spatial_downsample = spatial_downsample
        self.output_channels = output_channels

        self.convnet_1x1 = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=(1, 1),
            # spatial_downsample is unconstrained for 1x1 convolutions.
            stride=(spatial_downsample, spatial_downsample),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 5:
            # Timeseries, do it to each timestep independently
            outs = []
            for i in range(x.shape[1]):
                outs.append(self.convnet_1x1(x[:, i, :, :, :]))
            x = torch.stack(outs, dim=1)
        else:
            x = self.convnet_1x1(x)

        return x


class ImageEncoderPatches(torch.nn.Module):
    def __init__(
        self, spatial_downsample: int = 4, temporal_downsample: int = 1,
    ):
        """
        Image encoder that uses patches

        Args:
            spatial_downsample: How much to downsample spatially
            temporal_downsample: How much to downsample temporally
        """
        super().__init__()
        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = space_to_depth(
            x,
            temporal_block_size=self.temporal_downsample,
            spatial_block_size=self.spatial_downsample,
        )

        # For flow
        if x.ndim == 5 and x.shape[1] == 1:
            x = x.squeeze(axis=1)

        return x


class ImageEncoderPixel(torch.nn.Module):
    def __init__(
        self, spatial_downsample: int = 4, temporal_downsample: int = 1,
    ):
        """
        Image encoder class for simple downsampling with pixels

        Args:
            spatial_downsample: How much to downsample spatially
            temporal_downsample: How much to downsample temporally
        """
        super().__init__()
        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class ImageEncoderMetNet(nn.Module):
    def __init__(
        self, crop_size: int = 256, use_space2depth: bool = True,
    ):
        """
        Performs the MetNet preprocessing of mean pooling Sat channels, followed by
        concatenating the center crop and mean pool

        In the paper, the radar data is space2depth'd, while satellite channel is mean pooled, but for this different
        task, we choose to do either option for satellites

        Args:
            sat_channels: Number of satellite channels
            crop_size: Center crop size
            use_space2depth: Whether to use space2depth on satellite channels, or mean pooling, like in paper
        """
        super().__init__()
        # Split off sat + mask channels into own image, and the rest, which we just take a center crop
        # For this,
        self.sat_downsample = (
            torch.nn.PixelUnshuffle(downscale_factor=2)
            if use_space2depth
            else torch.nn.AvgPool3d(kernel_size=(1, 2, 2))
        )
        self.center_crop = torchvision.transforms.CenterCrop(size=crop_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sat_downsample(x)
        # In paper, satellite and radar data is concatenated here
        # We are just going to skip that bit
        sat_center = self.center_crop(x)
        sat_mean = F.avg_pool3d(x, (1, 2, 2))
        # All the same size now, so concatenate together, already have time, lat/long, and elevation image
        x = torch.cat([sat_center, sat_mean], dim=2)
        return x
