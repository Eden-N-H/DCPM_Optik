"""Depthwise Separable Convolution module."""
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise conv + pointwise conv.

    Factorizes a standard convolution into a depthwise convolution (spatial filtering
    per channel) followed by a pointwise 1x1 convolution (channel mixing). This reduces
    parameters from (C_in * C_out * K^2) to (C_in * K^2 + C_in * C_out).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            dilation=dilation, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
