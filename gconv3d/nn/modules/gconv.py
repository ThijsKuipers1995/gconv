"""
gconv.py

Implements group convolution base models.
"""
from __future__ import annotations

from gconv3d.nn.kernels import (
    GroupKernel,
    GLiftingKernel,
    GSeparableKernel,
    GSubgroupKernel,
    GKernel,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import math

from torch import Tensor

from torch.nn.modules.utils import _reverse_repeat_tuple, _pair, _triple


class GroupConvNd(nn.Module):
    def __init__(
        self,
        kernel: GroupKernel,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        conv_mode: str = "3d",
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.kernel = kernel

        # just for easy access
        self.in_channels = kernel.in_channels
        self.out_channels = kernel.out_channels
        self.kernel_size = kernel.kernel_size
        self.group_kernel_size = kernel.group_kernel_size
        self.groups = kernel.groups

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.padding_mode = padding_mode

        # set padding settings
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel.kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    dilation,
                    kernel.kernel_size,
                    range(len(kernel.kernel_size) - 1, -1, -1),
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
                self.padding, 2
            )

        if conv_mode == "2d":
            self._conv_forward = self._conv2d_forward
            bias_shape = (1, 1, 1)
        elif conv_mode == "3d":
            self._conv_forward = self._conv3d_forward
            bias_shape = (1, 1, 1, 1)

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, *bias_shape))

        self.reset_parameters()

    def reset_parameters(self):
        self.kernel.reset_parameters()

        if self.bias is None:
            return
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.kernel.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _conv2d_forward(self, input: Tensor, weight: Tensor, groups: int):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                None,
                self.stride,
                _pair(0),
                self.dilation,
                groups,
            )
        return F.conv2d(
            input, weight, None, self.stride, self.padding, self.dilation, groups
        )

    def _conv3d_forward(self, input: Tensor, weight: Tensor, groups: int):
        if self.padding_mode != "zeros":
            return F.conv3d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                None,
                self.stride,
                _triple(0),
                self.dilation,
                groups,
            )
        return F.conv3d(
            input, weight, None, self.stride, self.padding, self.dilation, groups
        )


class GLiftingConvNd(GroupConvNd):
    def __init__(
        self,
        kernel: GLiftingKernel,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        conv_mode: str = "3d",
        bias: bool = False,
    ) -> None:
        super().__init__(
            kernel, stride, padding, dilation, padding_mode, conv_mode, bias
        )

    def forward(self, input: Tensor, H: Tensor) -> tuple[Tensor, Tensor]:
        N = input.shape[0]
        num_out_H = H.shape[0]

        weight = self.kernel(H)

        input = self._conv_forward(
            input,
            weight.reshape(-1, self.in_channels // self.groups, *self.kernel_size),
            num_out_H,
            self.groups,
        ).view(N, self.out_channels, num_out_H, *input.shape[2:])

        input = input if self.bias is None else input + self.bias

        return input, H


class GLiftingConv2d(GLiftingConvNd):
    def __init__(
        self,
        kernel: GLiftingKernel,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = False,
    ) -> None:
        super().__init__(
            kernel, stride, padding, dilation, padding_mode, conv_mode="2d", bias=bias
        )


class GLiftingConv3d(GLiftingConvNd):
    def __init__(
        self,
        kernel: GLiftingKernel,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = False,
    ) -> None:
        super().__init__(
            kernel, stride, padding, dilation, padding_mode, conv_mode="3d", bias=bias
        )


class GSeparableConvNd(GroupConvNd):
    def __init__(
        self,
        kernel: GSeparableKernel,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        conv_mode: str = "3d",
        bias: bool = False,
    ) -> None:
        super().__init__(
            kernel, stride, padding, dilation, padding_mode, conv_mode, bias
        )

    def forward(
        self, input: Tensor, in_H: Tensor, out_H: Tensor
    ) -> tuple[Tensor, Tensor]:
        N, _, _, *input_dims = input.shape
        num_in_H, num_out_H = in_H.shape[0], out_H.shape[0]

        weight_H, weight_Rn = self.kernel(in_H, out_H)

        # subgroup conv
        input = self._conv_forward(
            input.reshape(N, self.in_channels * num_in_H, *input_dims),
            weight_H.reshape(
                self.out_channels * num_out_H,
                (self.in_channels // self.groups) * num_in_H,
                *self.kernel_size,
            ),
            self.groups,
        )

        # spatial conv
        input = self._conv_forward(
            input, weight_Rn, self.out_channels * num_out_H
        ).view(N, self.out_channels, num_out_H, *input.shape[2:])

        input = input if self.bias is None else input + self.bias

        return input, out_H


class GSeparableConv2d(GSeparableConvNd):
    def __init__(
        self,
        kernel: GSeparableKernel,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = False,
    ) -> None:
        super().__init__(kernel, stride, padding, dilation, padding_mode, "2d", bias)


class GSeparableConv3d(GSeparableConvNd):
    def __init__(
        self,
        kernel: GSeparableKernel,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = False,
    ) -> None:
        super().__init__(kernel, stride, padding, dilation, padding_mode, "3d", bias)


class GConvNd(GroupConvNd):
    def __init__(
        self,
        kernel: GKernel | GSubgroupKernel,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        conv_mode: str = "3d",
        bias: bool = False,
    ) -> None:
        super().__init__(
            kernel, stride, padding, dilation, padding_mode, conv_mode, bias
        )

    def forward(
        self, input: Tensor, in_H: Tensor, out_H: Tensor
    ) -> tuple[Tensor, Tensor]:
        N, _, _, *input_dims = input.shape
        num_in_H, num_out_H = in_H.shape[0], out_H.shape[0]

        weight = self.kernel(in_H, out_H)

        input = self._conv_forward(
            input.reshape(N, self.in_channels * num_in_H, *input_dims),
            weight.reshape(
                self.out_channels * num_out_H,
                (self.in_channels // self.groups) * num_in_H,
                *self.kernel_size,
            ),
            self.groups,
        ).view(N, self.out_channels, num_out_H, input.shape[2:])

        input = input if self.bias is None else input + self.bias

        return input, out_H


class GConv2d(GConvNd):
    def __init__(
        self,
        kernel: GKernel | GSubgroupKernel,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = False,
    ) -> None:
        super().__init__(kernel, stride, padding, dilation, padding_mode, "2d", bias)


class GConv3d(GConvNd):
    def __init__(
        self,
        kernel: GKernel | GSubgroupKernel,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = False,
    ) -> None:
        super().__init__(kernel, stride, padding, dilation, padding_mode, "3d", bias)
