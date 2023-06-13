"""
kernel.py

Contains base classes for group kernel, separable group kernel, and
lifting kernels.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init

from torch import Tensor

import math


class _Kernel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        groups: int = 1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size

        self.groups = groups


class GKernel(_Kernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        grid_H: Tensor,
        grid_Rn: Tensor,
        groups: int = 1,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, groups=groups)

        self.register_buffer("grid_H", grid_H)
        self.register_buffer("grid_Rn", grid_Rn)
        self.num_H = grid_H[0]

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, self.num_H, *kernel_size)
        )

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))


class GSeparableKernel(_Kernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        grid_H: Tensor,
        grid_Rn: Tensor,
        groups: int = 1,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, grid_H, groups=groups)

        self.register_buffer("grid_H", grid_H)
        self.register_buffer("grid_Rn", grid_Rn)

        self.num_H = grid_H[0]

        self.weight_H = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, self.num_H)
        )

        self.weight_Rn = nn.Parameter(torch.empty(out_channels, 1, kernel_size))

        init.kaiming_uniform_(self.weight_H, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_Rn, a=math.sqrt(5))


class GLiftingKernel(_Kernel):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        grid_Rn,
        groups: int = 1,
    ):
        super().__init__(in_channels, out_channels, kernel_size, groups)

        self.register_buffer("grid_Rn", grid_Rn)

        self.weight = torch.nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )

        init.kaiming_normal_(self.weight, a=math.sqrt(5))
