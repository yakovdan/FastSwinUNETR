from __future__ import annotations
from torch import nn
from typing import Sequence

import torch

class UnetrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """
        super().__init__()
        assert spatial_dims == 3
        self.conv_transp_3d = nn.ConvTranspose3d(in_channels,
                                                 out_channels,
                                                 kernel_size=upsample_kernel_size,
                                                 stride=upsample_kernel_size,
                                                 padding=0,
                                                 output_padding=0,
                                                 groups=1,
                                                 bias=False,
                                                 dilation=1)

        self.conv1 = nn.Conv3d(
            in_channels=96,
            out_channels=48,
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


    #inp shape: torch.Size([8, 48, 48, 48, 16])
    #skip shape: torch.Size([8, 48, 96, 96, 32])

    def forward(self, inp, skip):
        out = self.conv_transp_3d(inp)
        out = torch.cat((out, skip), dim=1)
        residual = out
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        residual = self.conv3(residual)
        residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


