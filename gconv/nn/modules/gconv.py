import torch
import torch.nn as nn

from gconv.nn import kernels
from torch.nn import init


class _GConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple | int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        permute_output_grid: bool = False,
    ) -> None:
        """
        Base class for group convolution.

        Arguments:
            - in_channels: Number of input channels.
            - out_channels: Number of output channels.
            - kernel_size: Spatial kernel size.
            - stride: Int denoting stride.
            - padding: Int denoting padding.
            - dilation: Int denoting dilation.
            - groups: Number of groups that have separate weights (channel-wise),
                      see Pytorch ConvNd documentation for details.
            - permute_output_grid: If True, will randomly permute output grid.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if type(kernel_size) is tuple
            else (kernel_size, kernel_size, kernel_size)
        )

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.permute_output_grid = permute_output_grid

    def extra_repr(self) -> str:
        s = ""
        s += f"{self.in_channels=}"
        s += f", {self.out_channels=}"
        s += f", bias={self.bias is not None}"
        s += f", {self.permute_output_grid=}"
        return s
