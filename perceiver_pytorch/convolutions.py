import torch
from torch.nn import functional as F


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
            layers += [
                self.make_layer(output_channels, output_channels, batch=use_batchnorm)
            ]

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
            padding=(1, 1),
        )
        self.transpose_conv2 = torch.nn.ConvTranspose2d(
            in_channels=output_channels * 2,
            out_channels=output_channels,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1),
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
            output_channels: Final output channels
            num_temporal_upsamples: Number of temporal upsamples to perform
            num_space_upsamples: Number of spatial upsamples to perform
        """

        super().__init__()
        temporal_stride = 2
        space_stride = 2
        num_upsamples = max(num_space_upsamples, num_temporal_upsamples)

        # create the input and output changesl for the different layers
        # The intermediate channels are (output channels) * 2^(number of upsamples - 1 - index of the current upsample)
        # The decoder sets the number of upsamples as log2(upsample_value), and this changes the number of channels
        # in a similar way, so it all scales together.
        intermediate_output_channels = [
            output_channels * pow(2, num_upsamples - 1 - i)
            for i in range(0, num_upsamples)
        ]
        intermediate_input_channels = [input_channels] + intermediate_output_channels

        self.layers = torch.nn.ModuleList()
        for i in range(num_upsamples):
            if i >= num_temporal_upsamples:
                temporal_stride = 1
            if i >= num_space_upsamples:
                space_stride = 1

            input_channels = input_channels if i == 0 else output_channels
            stride = (temporal_stride, space_stride, space_stride)
            conv = torch.nn.ConvTranspose3d(
                in_channels=intermediate_input_channels[i],
                out_channels=intermediate_output_channels[i],
                stride=stride,
                kernel_size=stride,
            )
            # see output dims calculations - https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html
            # if kernel=stride, and dilation -1
            # D = (D-1) * stride + (kernel - 1) + 2 = D * stride

            self.layers.append(conv)
            if i != num_upsamples - 1:
                self.layers.append(torch.nn.ReLU())

    def forward(self, x):
        # Move around to Channels first
        x = x.permute(0, 2, 1, 3, 4)

        # loop over layers
        for layer in self.layers:
            x = layer(x)

        x = F.relu(x)
        # Move back
        x = x.permute(0, 2, 1, 3, 4)
        return x
