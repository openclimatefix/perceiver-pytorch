import torch
import numpy as np
from perceiver_pytorch.utils import reverse_space_to_depth
from perceiver_pytorch.convolutions import Conv2DUpsample, Conv3DUpsample


class ImageDecoder(torch.nn.Module):
    def __init__(
        self,
        postprocess_type: str = "pixels",
        spatial_upsample: int = 1,
        temporal_upsample: int = 1,
        output_channels: int = -1,
        input_channels: int = 12,
        input_reshape_size=None,
    ):
        """
        ImageDecoder modeled after JAX version here
        https://github.com/deepmind/deepmind-research/blob/769bfdbeafbcb472cb8e2c6cfa746b53ac82efc2/perceiver/io_processors.py#L441-L510

        Args:
            postprocess_type: Type of postprocessing, one of conv, patches, pixels, raft, or conv1x1
            spatial_upsample: How much to spatially upsample
            temporal_upsample: How much to temporally upsample
            output_channels: Number of output channels, should be the final desired number of channels
                Has to explicitly set for conv and conv1x1 options, otherwise an error will be raised.
                Ignored for patches and pixels options.
            input_channels: Number of input channels to decoder
            input_reshape_size: The size to reshape the input to
        """

        super().__init__()

        if postprocess_type not in ("conv", "patches", "pixels", "conv1x1"):
            # TODO Add Raft
            raise ValueError("Invalid postprocess_type!")

        # Architecture parameters:
        self.postprocess_type = postprocess_type

        self.temporal_upsample = temporal_upsample
        self.spatial_upsample = spatial_upsample
        self.input_reshape_size = input_reshape_size

        if postprocess_type == "pixels":
            # No postprocessing for pixels
            self.decoder = torch.nn.Identity()
        elif postprocess_type == "patches":
            self.decoder = ImageDecoderPatches(
                spatial_upsample=spatial_upsample, temporal_upsample=temporal_upsample
            )
        elif postprocess_type == "conv":
            self.decoder = ImageDecoderConv(
                spatial_upsample=spatial_upsample,
                temporal_upsample=temporal_upsample,
                output_channels=output_channels,
                input_channels=input_channels,
            )
        elif postprocess_type == "conv1x1":
            self.decoder = ImageDecoderConv1x1(
                spatial_upsample=spatial_upsample,
                output_channels=output_channels,
                input_channels=input_channels,
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.input_reshape_size is not None:
            inputs = torch.reshape(
                inputs,
                [inputs.shape[0]] + list(self.input_reshape_size) + [inputs.shape[-1]],
            )
        return self.decoder(inputs)


class ImageDecoderConv(torch.nn.Module):
    def __init__(
        self,
        spatial_upsample: int = 1,
        temporal_upsample: int = 1,
        output_channels: int = -1,
        input_channels: int = 12,
    ):
        """
        Convolutional image decoder that can upsample temporally and spatially

        Args:
            spatial_upsample: How much to spatially upsample
            temporal_upsample: How much to temporally upsample
            output_channels: Number of output channels, should be the final desired number of channels
                Has to explicitly set for conv and conv1x1 options, otherwise an error will be raised.
                Ignored for patches and pixels options.
            input_channels: Number of input channels to decoder
        """

        super().__init__()

        self.temporal_upsample = temporal_upsample
        self.spatial_upsample = spatial_upsample

        if output_channels == -1:
            raise ValueError("Expected value for output_channels")
        if self.temporal_upsample != 1:

            def int_log2(x):
                return int(np.round(np.log(x) / np.log(2)))

            self.convnet = Conv3DUpsample(
                input_channels=input_channels,
                output_channels=output_channels,
                num_temporal_upsamples=int_log2(temporal_upsample),
                num_space_upsamples=int_log2(spatial_upsample),
            )
        else:
            assert (
                self.spatial_upsample == 4
            ), "Conv2DUpsample only support 4x spatial upsample right now"
            self.convnet = Conv2DUpsample(
                input_channels=input_channels, output_channels=output_channels
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Convnet image featurization.
        if len(inputs.shape) == 5 and self.temporal_upsample == 1:
            # Timeseries, do it to each timestep independently
            outs = []
            for i in range(inputs.shape[1]):
                outs.append(self.convnet(inputs[:, i, :, :, :]))
            inputs = torch.stack(outs, dim=1)
        else:
            inputs = self.convnet(inputs)

        return inputs


class ImageDecoderConv1x1(torch.nn.Module):
    def __init__(
        self,
        spatial_upsample: int = 1,
        output_channels: int = -1,
        input_channels: int = 12,
    ):
        """
        Convolutional 1x1 image decoder

        Args:
            spatial_upsample: How much to spatially upsample
            output_channels: Number of output channels, should be the final desired number of channels
                Has to explicitly set for conv and conv1x1 options, otherwise an error will be raised.
                Ignored for patches and pixels options.
            input_channels: Number of input channels to decoder
        """

        super().__init__()

        self.spatial_upsample = spatial_upsample

        if output_channels == -1:
            raise ValueError("Expected value for output_channels")
        self.conv1x1 = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=(1, 1),
            # spatial_downsample is unconstrained for 1x1 convolutions.
            stride=(self.spatial_upsample, self.spatial_upsample),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Convnet image featurization.
        if len(inputs.shape) == 5:
            # Timeseries, do it to each timestep independently
            outs = []
            for i in range(inputs.shape[1]):
                outs.append(self.conv1x1(inputs[:, i, :, :, :]))
            inputs = torch.stack(outs, dim=1)
        else:
            inputs = self.conv1x1(inputs)

        return inputs


class ImageDecoderPatches(torch.nn.Module):
    def __init__(
        self, spatial_upsample: int = 1, temporal_upsample: int = 1,
    ):
        """
        Patch-based image decoder

        Args:
            spatial_upsample: How much to spatially upsample
            temporal_upsample: How much to temporally upsample
        """

        super().__init__()

        self.temporal_upsample = temporal_upsample
        self.spatial_upsample = spatial_upsample

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = reverse_space_to_depth(
            inputs, self.temporal_upsample, self.spatial_upsample
        )
        return inputs
