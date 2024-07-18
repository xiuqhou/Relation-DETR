import math
from typing import Tuple, Union

import torch
from torch import nn
from torchvision.ops import DeformConv2d


class DeformConv2dPack(nn.Module):
    """This is a pack of deformable convolution that can be used as normal convolution"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: Union[bool, str] = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (
                kernel_size,
                kernel_size,
            )
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.conv_offset = nn.Conv2d(
            in_channels,
            groups * 2 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,  # Don't know whether to add groups here
            bias=True,
        )
        self.conv_mask = nn.Conv2d(
            in_channels,
            groups * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True,
        )
        self.deform_conv2d = DeformConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.init_weights()

    def init_weights(self) -> None:
        self.conv_offset.weight.data.zero_()
        self.conv_mask.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        self.conv_mask.bias.data.zero_()
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.deform_conv2d.weight.data.uniform_(-stdv, stdv)
        if self.deform_conv2d.bias is not None:
            self.deform_conv2d.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x))
        out = self.deform_conv2d(x, offset, mask)
        return out
