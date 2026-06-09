from __future__ import annotations
from torch import nn
import torch
import numpy as np
from typing import Sequence
class UnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
    ):
        super().__init__()
        assert stride == 1
        assert in_channels == 48
        assert out_channels == 48
        assert kernel_size == 3
        assert norm_name == "instance"
        assert act_name == "leakyrelu"
        assert dropout is None

        self.conv1 = nn.Conv3d(
            in_channels = 96,
            out_channels = 48,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.conv2 = nn.Conv3d(
            in_channels=48,
            out_channels=48,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.norm1 = nn.InstanceNorm3d(num_features=48)
        self.norm2 = nn.InstanceNorm3d(num_features=48)
        self.downsample = in_channels != out_channels
        self.conv3 = nn.Conv3d(in_channels=48,
                               out_channels=48,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=1,
                               bias=False)

        self.norm3 = nn.InstanceNorm3d(num_features=48)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        residual = self.conv3(residual)
        residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out