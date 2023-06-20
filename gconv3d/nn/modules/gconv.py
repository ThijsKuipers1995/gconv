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
    __constants__ = ["stride", "padding", "dilation", "padding_mode"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        group_kernel_size: int | tuple,
        kernel: GroupKernel,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",  # NOTE: I like it this way
        conv_mode: str = "3d",
        bias: bool = False,
    ) -> None:
        super().__init__()

        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings
                    )
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )

        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                "padding_mode must be one of {}, but got padding_mode='{}'".format(
                    valid_padding_modes, padding_mode
                )
            )

        self.kernel = kernel

        # just for easy access
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_kernel_size = group_kernel_size
        self.groups = groups

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.padding_mode = padding_mode

        if conv_mode == "2d":
            self._conv_forward = self._conv2d_forward
            bias_shape = (1, 1, 1)
        elif conv_mode == "3d":
            self._conv_forward = self._conv3d_forward
            bias_shape = (1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unspported conv mode: got {conv_mode=}, expected `2d` or `3d`."
            )

        # init padding settings
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

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, *bias_shape))
        else:
            self.register_buffer("bias", None)

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
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        group_kernel_size: int,
        kernel: GroupKernel,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        conv_mode: str = "3d",
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size,
            kernel,
            groups,
            stride,
            padding,
            dilation,
            padding_mode,
            conv_mode,
            bias,
        )

    def forward(self, input: Tensor, H: Tensor) -> tuple[Tensor, Tensor]:
        N = input.shape[0]
        num_out_H = H.shape[0]

        weight = self.kernel(H)

        input = self._conv_forward(
            input,
            weight.reshape(-1, self.in_channels // self.groups, *self.kernel_size),
            self.groups,
        ).view(N, self.out_channels, num_out_H, *input.shape[2:])

        if self.bias is not None:
            input = input + self.bias

        return input, H


class GSeparableConvNd(GroupConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        group_kernel_size: int,
        kernel: GroupKernel,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        conv_mode: str = "3d",
        bias: bool = False,
    ) -> None:
        """
        Implementation the Nd separable group convolution.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size,
            kernel,
            groups,
            stride,
            padding,
            dilation,
            padding_mode,
            conv_mode,
            bias,
        )

    def forward(
        self, input: Tensor, in_H: Tensor, out_H: Tensor
    ) -> tuple[Tensor, Tensor]:
        N, _, _, *input_dims = input.shape
        num_in_H, num_out_H = in_H.shape[0], out_H.shape[0]

        weight_H, weight = self.kernel(in_H, out_H)

        # subgroup conv
        input = self._conv_forward(
            input.reshape(N, self.in_channels * num_in_H, *input_dims),
            weight_H.reshape(
                self.out_channels * num_out_H,
                (self.in_channels // self.groups) * num_in_H,
                *weight_H.shape[4:],
            ),
            self.groups,
        )

        # spatial conv
        input = self._conv_forward(
            input,
            weight.reshape(
                self.out_channels * num_out_H,
                1,
                *self.kernel_size,
            ),
            self.out_channels * num_out_H,
        ).view(N, self.out_channels, num_out_H, *input.shape[2:])

        if self.bias is not None:
            input = input + self.bias

        return input, out_H


class GConvNd(GroupConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        group_kernel_size: int,
        kernel: GroupKernel,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        conv_mode: str = "3d",
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size,
            kernel,
            groups,
            stride,
            padding,
            dilation,
            padding_mode,
            conv_mode,
            bias,
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
        ).view(N, self.out_channels, num_out_H, *input.shape[3:])

        if self.bias is not None:
            input = input + self.bias

        return input, out_H


class GLiftingConv2d(GLiftingConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int | tuple,
        kernel: GLiftingKernel,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = False,
    ) -> None:
        """
        Implements 2d lifting convolution.

        Arguments:
            - int_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: tuple denoting the spatial kernel size.
            - group_kernel_size: int or tuple denoting the group kernel size.
                                    In the case of a tuple, each element denotes
                                    a separate kernel size for each subgroup. For
                                    example, (4, 2) could denote a O3 kernel with
                                    rotation and reflection kernels of size 4 and
                                    2, respectively.
            - kernel: GroupKernel that manages the group and samples weights.
            - groups: int denoting the number of groups for depth-wise separability.
            - stride: int denoting the stride.
            - padding: int or denoting padding.
            - dilation: int denoting dilation.
            - padding_mode: str denoting the padding mode.
            - bias: bool that if true will initialzie bias parameters.
        """
        super().__init__(
            in_channels,
            out_channels,
            _pair(kernel_size),
            group_kernel_size,
            kernel,
            groups,
            _pair(stride),
            padding,
            _pair(dilation),
            padding_mode,
            conv_mode="2d",
            bias=bias,
        )


class GSeparableConv2d(GSeparableConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int | tuple,
        kernel: GSubgroupKernel,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = False,
    ) -> None:
        """
        Implements 2d separable group convolution.

        Arguments:
            - int_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: tuple denoting the spatial kernel size.
            - group_kernel_size: int or tuple denoting the group kernel size.
                                    In the case of a tuple, each element denotes
                                    a separate kernel size for each subgroup. For
                                    example, (4, 2) could denote a O3 kernel with
                                    rotation and reflection kernels of size 4 and
                                    2, respectively.
            - kernel: GroupKernel that manages the group and samples weights.
            - groups: int denoting the number of groups for depth-wise separability.
            - stride: int denoting the stride.
            - padding: int or denoting padding.
            - dilation: int denoting dilation.
            - padding_mode: str denoting the padding mode.
            - bias: bool that if true will initialzie bias parameters.
        """
        super().__init__(
            in_channels,
            out_channels,
            _pair(kernel_size),
            group_kernel_size,
            kernel,
            groups,
            _pair(stride),
            padding,
            _pair(dilation),
            padding_mode,
            conv_mode="2d",
            bias=bias,
        )


class GConv2d(GConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int | tuple,
        kernel: GKernel | GSubgroupKernel,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = False,
    ) -> None:
        """
        Implements 2d group convolution.

        Arguments:
            - int_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: tuple denoting the spatial kernel size.
            - group_kernel_size: int or tuple denoting the group kernel size.
                                    In the case of a tuple, each element denotes
                                    a separate kernel size for each subgroup. For
                                    example, (4, 2) could denote a O3 kernel with
                                    rotation and reflection kernels of size 4 and
                                    2, respectively.
            - kernel: GroupKernel that manages the group and samples weights.
            - groups: int denoting the number of groups for depth-wise separability.
            - stride: int denoting the stride.
            - padding: int or denoting padding.
            - dilation: int denoting dilation.
            - padding_mode: str denoting the padding mode.
            - bias: bool that if true will initialzie bias parameters.
        """
        super().__init__(
            in_channels,
            out_channels,
            _pair(kernel_size),
            group_kernel_size,
            kernel,
            groups,
            _pair(stride),
            padding,
            _pair(dilation),
            padding_mode,
            conv_mode="2d",
            bias=bias,
        )


class GLiftingConv3d(GLiftingConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int,
        kernel: GLiftingKernel,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = False,
    ) -> None:
        """
        Implements 3d lifting convolution.

        Arguments:
            - int_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: tuple denoting the spatial kernel size.
            - group_kernel_size: int or tuple denoting the group kernel size.
                                    In the case of a tuple, each element denotes
                                    a separate kernel size for each subgroup. For
                                    example, (4, 2) could denote a O3 kernel with
                                    rotation and reflection kernels of size 4 and
                                    2, respectively.
            - kernel: GroupKernel that manages the group and samples weights.
            - groups: int denoting the number of groups for depth-wise separability.
            - stride: int denoting the stride.
            - padding: int or denoting padding.
            - dilation: int denoting dilation.
            - padding_mode: str denoting the padding mode.
            - bias: bool that if true will initialzie bias parameters.
        """
        super().__init__(
            in_channels,
            out_channels,
            _triple(kernel_size),
            group_kernel_size,
            kernel,
            groups,
            _triple(stride),
            padding,
            _triple(dilation),
            padding_mode,
            conv_mode="3d",
            bias=bias,
        )


class GSeparableConv3d(GSeparableConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int,
        kernel: GSeparableKernel,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = False,
    ) -> None:
        """
        Implements 3d separable group convolution.

        Arguments:
            - int_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: tuple denoting the spatial kernel size.
            - group_kernel_size: int or tuple denoting the group kernel size.
                                    In the case of a tuple, each element denotes
                                    a separate kernel size for each subgroup. For
                                    example, (4, 2) could denote a O3 kernel with
                                    rotation and reflection kernels of size 4 and
                                    2, respectively.
            - kernel: GroupKernel that manages the group and samples weights.
            - groups: int denoting the number of groups for depth-wise separability.
            - stride: int denoting the stride.
            - padding: int or denoting padding.
            - dilation: int denoting dilation.
            - padding_mode: str denoting the padding mode.
            - bias: bool that if true will initialzie bias parameters.
        """
        super().__init__(
            in_channels,
            out_channels,
            _triple(kernel_size),
            group_kernel_size,
            kernel,
            groups,
            _triple(stride),
            padding,
            _triple(dilation),
            padding_mode,
            conv_mode="3d",
            bias=bias,
        )


class GConv3d(GConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int,
        kernel: GKernel | GSubgroupKernel,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = False,
    ) -> None:
        """
        Implements 3d group convolution.

        Arguments:
            - int_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: tuple denoting the spatial kernel size.
            - group_kernel_size: int or tuple denoting the group kernel size.
                                    In the case of a tuple, each element denotes
                                    a separate kernel size for each subgroup. For
                                    example, (4, 2) could denote a O3 kernel with
                                    rotation and reflection kernels of size 4 and
                                    2, respectively.
            - kernel: GroupKernel that manages the group and samples weights.
            - groups: int denoting the number of groups for depth-wise separability.
            - stride: int denoting the stride.
            - padding: int or denoting padding.
            - dilation: int denoting dilation.
            - padding_mode: str denoting the padding mode.
            - bias: bool that if true will initialzie bias parameters.
        """
        super().__init__(
            in_channels,
            out_channels,
            _triple(kernel_size),
            group_kernel_size,
            kernel,
            groups,
            _triple(stride),
            padding,
            _triple(dilation),
            padding_mode,
            conv_mode="3d",
            bias=bias,
        )
