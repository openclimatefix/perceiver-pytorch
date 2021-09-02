import torch
import torch.nn.functional as F
import numpy as np
import math
import einops

def extract_image_patches(x, kernel, stride=1, dilation=1):
    # Do TF 'SAME' Padding
    b, c, h, w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(x, (pad_row // 2, pad_row - pad_row // 2, pad_col // 2, pad_col - pad_col // 2))

    # Extract patches
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()

    return patches.view(b, -1, patches.shape[-2], patches.shape[-1])


def reverse_space_to_depth(
        frames: np.ndarray, temporal_block_size: int = 1, spatial_block_size: int = 1
) -> np.ndarray:
    """Reverse space to depth transform."""
    if len(frames.shape) == 4:
        return einops.rearrange(
            frames,
            "b h w (dh dw c) -> b (h dh) (w dw) c",
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    elif len(frames.shape) == 5:
        return einops.rearrange(
            frames,
            "b t h w (dt dh dw c) -> b (t dt) (h dh) (w dw) c",
            dt=temporal_block_size,
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    else:
        raise ValueError(
            "Frames should be of rank 4 (batch, height, width, channels)"
            " or rank 5 (batch, time, height, width, channels)"
        )


def space_to_depth(
        frames: np.ndarray, temporal_block_size: int = 1, spatial_block_size: int = 1
) -> np.ndarray:
    """Space to depth transform."""
    if len(frames.shape) == 4:
        return einops.rearrange(
            frames,
            "b (h dh) (w dw) c -> b h w (dh dw c)",
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    elif len(frames.shape) == 5:
        return einops.rearrange(
            frames,
            "b (t dt) (h dh) (w dw) c -> b t h w (dt dh dw c)",
            dt=temporal_block_size,
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    else:
        raise ValueError(
            "Frames should be of rank 4 (batch, height, width, channels)"
            " or rank 5 (batch, time, height, width, channels)"
        )


class Conv2DDownsample(torch.nn.Sequential):
    def __init__(
            self,
            num_layers: int = 1,
            input_channels: int = 12,
            output_channels: int = 64,
            use_batchnorm: bool = True,
    ):
        """
        Constructs a Conv2DDownsample model
        Args:
            num_layers: Number of conv -> maxpool layers
            output_channels: Number of output channels
            input_channels: Number of input channels to first layer
            use_batchnorm: Whether to use Batch Norm
        """

        layers = [self.make_layer(input_channels, output_channels, batch=use_batchnorm)]
        for _ in range(num_layers - 1):
            layers += [self.make_layer(output_channels, output_channels, batch=use_batchnorm)]

        super().__init__(*layers)

    def make_layer(self, c_in, c_out, ks=7, stride=2, padding=7 // 2, batch=True):
        "Make Conv->Batch->Relu->MaxPool stack"
        layers = [torch.nn.Conv2d(c_in, c_out, ks, stride, padding, bias=False)]
        if batch:
            layers += [torch.nn.BatchNorm2d(c_out)]
        layers += [torch.nn.ReLU(), torch.nn.MaxPool2d(3, stride=2, padding=3 // 2)]
        return torch.nn.Sequential(*layers)


class Conv2DUpsample(torch.nn.Module):
    def __init__(self, input_channels: int = 12, output_channels: int = 12):
        """
        Upsamples 4x using 2 2D transposed convolutions
        Args:
            input_channels: Input channels to the first layer
            output_channels: Number of output channels
        """

        super().__init__()
        self.transpose_conv1 = torch.nn.ConvTranspose2d(
            in_channels=input_channels,
            out_channels=output_channels * 2,
            kernel_size=(4, 4),
            stride=(2, 2),
        )
        self.transpose_conv2 = torch.nn.ConvTranspose2d(
            in_channels=output_channels * 2,
            out_channels=output_channels,
            kernel_size=(4, 4),
            stride=(2, 2),
        )

    def forward(self, x):
        x = self.transpose_conv1(x)
        x = F.relu(x)
        x = self.transpose_conv2(x)
        return x


class Conv3DUpsample(torch.nn.Module):
    def __init__(
            self,
            input_channels: int = 12,
            output_channels: int = 12,
            num_temporal_upsamples: int = 2,
            num_space_upsamples: int = 4,
    ):
        """
        Simple convolutional auto-encoder
        Args:
            output_channels:
            num_temporal_upsamples:
            num_space_upsamples:
        """

        super().__init__()
        temporal_stride = 2
        space_stride = 2
        num_upsamples = max(num_space_upsamples, num_temporal_upsamples)
        self.layers = torch.nn.ModuleList()
        for i in range(num_upsamples):
            if i >= num_temporal_upsamples:
                temporal_stride = 1
            if i >= num_space_upsamples:
                space_stride = 1

            channels = output_channels * pow(2, num_upsamples - 1 - i)
            conv = torch.nn.ConvTranspose3d(
                in_channels=input_channels,
                out_channels=channels,
                stride=(temporal_stride, space_stride, space_stride),
                kernel_size=(4, 4, 4),
            )
            self.layers.append(conv)
            if i != num_upsamples - i:
                self.layers.append(torch.nn.ReLU())

    def forward(self, x):
        return self.layers.forward(x)


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
                x = x[:, :: self.spatial_downsample, :: self.spatial_downsample]
            elif x.ndim == 5:
                x = x[
                    :,
                    :: self.temporal_downsample,
                    :: self.spatial_downsample,
                    :: self.spatial_downsample,
                    ]
            else:
                raise ValueError("Unsupported data format for pixels")

        return x


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
        super().__init__()

        if postprocess_type not in ("conv", "patches", "pixels", "raft", "conv1x1"):
            raise ValueError("Invalid postproc_type!")

        # Architecture parameters:
        self.postprocess_type = postprocess_type

        self.temporal_upsample = temporal_upsample
        self.spatial_upsample = spatial_upsample
        self.input_reshape_size = input_reshape_size

        if self.postprocess_type == "pixels":
            # No postprocessing.
            if self.temporal_upsample != 1 or self.spatial_upsample != 1:
                raise ValueError("Pixels postprocessing should not currently upsample.")
        elif self.postprocess_type == "conv1x1":
            assert self._temporal_upsample == 1, "conv1x1 does not upsample in time."
            if output_channels == -1:
                raise ValueError("Expected value for n_outputs")
            self.conv1x1 = torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(1, 1),
                # spatial_downsample is unconstrained for 1x1 convolutions.
                stride=(self.spatial_upsample, self.spatial_upsample),
            )
        elif self.postprocess_type == "conv":
            if output_channels == -1:
                raise ValueError("Expected value for n_outputs")
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
                self.convnet = Conv2DUpsample(
                    input_channels=input_channels, output_channels=output_channels
                )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.input_reshape_size is not None:
            inputs = torch.reshape(
                inputs, [inputs.shape[0]] + list(self.input_reshape_size) + [inputs.shape[-1]]
            )

        if self.postprocess_type == "conv" or self.postprocess_type == "raft":
            # Convnet image featurization.
            if len(inputs.shape) == 5 and self.temporal_upsample == 1:
                # Timeseries, do it to each timestep independently
                outs = []
                for i in range(inputs.shape[1]):
                    outs.append(self.convnet(inputs[:, i, :, :, :]))
                inputs = torch.stack(outs, dim=1)
            else:
                inputs = self.convnet(inputs)
        elif self.postprocess_type == "conv1x1":
            inputs = self.conv1x1(inputs)
        elif self.postprocess_type == "patches":
            inputs = reverse_space_to_depth(inputs, self.temporal_upsample, self.spatial_upsample)

        return inputs
