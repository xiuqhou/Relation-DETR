from functools import partial
from typing import List

from torch import nn
from models.bricks.misc import Conv2dNormActivation


class ChannelMapper(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        norm_layer=partial(nn.GroupNorm, 32),
        activation_layer: nn.Module = None,
        dilation: int = 1,
        inplace: bool = True,
        bias: bool = None,
    ):
        self.in_channels = in_channels
        super().__init__()
        self.convs = nn.ModuleList()
        self.num_channels = [out_channels] * num_outs
        for in_channel in in_channels:
            self.convs.append(
                Conv2dNormActivation(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=bias,
                    groups=groups,
                    dilation=dilation,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    inplace=inplace,
                )
            )
        for _ in range(num_outs - len(in_channels)):
            self.convs.append(
                Conv2dNormActivation(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=bias,
                    groups=groups,
                    dilation=dilation,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    inplace=inplace,
                )
            )
            in_channel = out_channels

        self.init_weights()

    def init_weights(self):
        # initialize modules
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight, gain=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, inputs):
        """Map inputs into specific embed_dim. If the length of inputs is smaller
        than convolution modules, only the last `len(inputs) + extra` will be used
        for mapping.

        :param inputs: dict containing torch.Tensor.
        :return: a list of mapped torch.Tensor
        """
        inputs = list(inputs.values())
        assert len(inputs) <= len(self.in_channels)
        start = len(self.in_channels) - len(inputs)
        convs = self.convs[start:]
        outs = [convs[i](inputs[i]) for i in range(len(inputs))]
        for i in range(len(inputs), len(convs)):
            if i == len(inputs):
                outs.append(convs[i](inputs[-1]))
            else:
                outs.append(convs[i](outs[-1]))
        return outs
