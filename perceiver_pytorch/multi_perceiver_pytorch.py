from perceiver_pytorch.perceiver_io import PerceiverIO
from perceiver_pytorch.modalities import InputModality, modality_encoding
from perceiver_pytorch.utils import encode_position, fourier_encode
import torch
from typing import List, Iterable, Dict, Optional, Any, Union, Tuple
from einops import rearrange, repeat
from math import prod


class MultiPerceiver(torch.nn.Module):
    def __init__(
        self,
        modalities: Iterable[InputModality],
        fourier_encode_data: bool = True,
        input_channels: int = 3,
        output_channels: int = 12,
        forecast_steps: int = 48,
        sine_only: bool = False,
        output_shape: Union[int, Tuple[int, ...]] = 32,
        **kwargs,
    ):
        """
        PerceiverIO made to work more specifically with timeseries images and multimodal inputs https://arxiv.org/abs/2107.14795
        This is a wrapper around the PerceiverIO implementation to encode the inputs correctly

        Args:
            input_channels: Number of input channels (int)
            forecast_steps: Number of forecast steps to make (int)
            fourier_encode_data: Whether to add Fourier Features to the input data, if this is false, inputs should be have some type of positional encoding added beforehand
            output_channels: Number of output channels per image (int)
            sine_only: Only use Sine part of Fourier features (bool)
            output_shape: Int or Tuple of ints, giving the desired output shape of the model
            **kwargs: Extra kwargs to pass through to PerceiverIO
        """
        super(MultiPerceiver, self).__init__()
        self.fourier_encode_data = fourier_encode_data
        self.forecast_steps = forecast_steps
        self.input_channels = input_channels
        self.sine_only = sine_only
        self.output_channels = output_channels
        self.modalities = {modality.name: modality for modality in modalities}
        # we encode modality with one hot encoding, so need one dim per modality:
        modality_encoding_dim = len(modalities)
        # input_dim is the maximum dimension over all input modalities:
        input_dim = max(modality.input_dim for modality in modalities) + modality_encoding_dim
        # Pop dim
        self.max_modality_dim = input_dim
        kwargs.pop("dim", None)
        # Want toe logit_dim to be the same as the channels * width or height
        if isinstance(output_shape, int):
            kwargs["logits_dim"] = output_shape * self.output_channels
        else:
            kwargs["logits_dim"] = prod(output_shape)
        self.perceiver = PerceiverIO(dim=input_dim, **kwargs)

    def decode_output(self, data):
        pass

    def forward(self, multi_modality_data: Dict[str, torch.Tensor], mask=None, queries=None):
        batch_sizes = set()
        num_modalities = len(multi_modality_data)
        linearized_data = []

        for modality_index, modality_name in enumerate(sorted(multi_modality_data.keys())):
            assert (
                modality_name in self.modalities
            ), f"modality {modality_name} was not defined in constructor"
            data = multi_modality_data[modality_name]
            modality = self.modalities[modality_name]
            b, *axis, _ = data.size()
            assert len(axis) == modality.input_axis, (
                f"input data must have the right number of axes for modality {modality_name}. "
                f"Expected {modality.input_axis} while forward argument offered {len(axis)}"
            )
            batch_sizes.add(b)
            assert len(batch_sizes) == 1, "batch size must be the same across all modalities"
            enc_pos = []
            if self.fourier_encode_data:
                # calculate fourier encoded positions in the range of [-1, 1], for all axis
                enc_pos = encode_position(
                    batch_size=b,
                    axis=axis,
                    max_frequency=modality.max_freq,
                    num_frequency_bands=modality.num_freq_bands,
                    sine_only=self.sine_only,
                ).type_as(data)

            # Figure out padding for this modality, given max dimension across all modalities:
            padding_size = self.max_modality_dim - modality.input_dim - num_modalities

            padding = torch.zeros(size=data.size()[0:-1] + (padding_size,)).type_as(data)
            # concat to channels of data and flatten axis
            modality_encodings = modality_encoding(b, axis, modality_index, num_modalities).type_as(
                data
            )
            to_concat = (
                (data, padding, enc_pos, modality_encodings)
                if len(enc_pos) > 0
                else (data, padding, modality_encodings)
            )
            data = torch.cat(to_concat, dim=-1)
            # concat to channels of data and flatten axis
            data = rearrange(data, "b ... d -> b (...) d")
            linearized_data.append(data)

        # Concatenate all the modalities:
        data = torch.cat(linearized_data, dim=1)

        # After this is the PerceiverIO backbone, still would need to decode it back to an image though
        # Should include the query shape here for the output we want, could be learned embeddings, repeated input frames of the same shape that is desired, etc.
        perceiver_output = self.perceiver.forward(data, mask, queries)

        # For multiple modalities, they are split after this back into different tensors
        # For Sat images, we just want the images, not the other ones, so can leave it as is?

        # Have to decode back into future Sat image frames
        # Perceiver for 'pixel' postprocessing does nothing, or undoes the space2depth from before if just image
        # If doing depth2space, should split modalities again

        # Reshape to the correct output
        # This is how it is done in the official implementation, do a decoder query with cross attention, then just reshape the output
        # For timeseries, this is given as a query with T*H*W shape
        # For Flow Decoder, this is the same, except has a rescale factor

        # To keep this more general, leave the reshaping to postprocessing outside the model

        return perceiver_output
