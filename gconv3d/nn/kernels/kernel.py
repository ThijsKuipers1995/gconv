"""
kernel.py

Contains base classes for group kernel, separable group kernel, and
lifting kernels.
"""
from __future__ import annotations

from typing import Callable

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
        mask: Tensor | None = None,
        det_H: Callable | None = None,
        inverse_H: Callable | None = None,
        left_apply_to_H: Callable | None = None,
        left_apply_to_Rn: Callable | None = None,
        interpolate_H: Callable | None = None,
        interpolate_Rn: Callable | None = None,
        interpolate_H_kwargs: dict = {},
        interpolate_Rn_kwargs: dict = {},
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size

        self.groups = groups

        self.register_buffer("mask", mask)

        self.det_H = det_H
        self.inverse_H = inverse_H
        self.left_apply_to_H = left_apply_to_H
        self.left_apply_to_Rn = left_apply_to_Rn
        self.interpolate_H = interpolate_H
        self.interpolate_Rn = interpolate_Rn
        self.interpolate_H_kwargs = interpolate_H_kwargs
        self.interpolate_Rn_kwargs = interpolate_Rn_kwargs


class GLiftingKernel(_Kernel):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        grid_Rn,
        groups: int = 1,
        mask: Tensor | None = None,
        det_H: Callable | None = None,
        inverse_H: Callable | None = None,
        left_apply_to_Rn: Callable | None = None,
        interpolate_Rn: Callable | None = None,
        interpolate_Rn_kwargs: dict = {},
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            groups=groups,
            mask=mask,
            det_H=det_H,
            inverse_H=inverse_H,
            left_apply_to_Rn=left_apply_to_Rn,
            interpolate_Rn=interpolate_Rn,
            interpolate_Rn_kwargs=interpolate_Rn_kwargs,
        )

        self.register_buffer("_grid_Rn", grid_Rn)

        self.weight = torch.nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )

        init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, H):
        H_product = self.left_apply_to_Rn(self.inverse_H(H), self._grid_Rn)

        weight = self.interpolate_H(
            self.weight.repeat_interleave(H.shape[0], dim=0),
            H_product.repeat(self.out_channels, 1, 1, 1, 1),
            **self.interpolate_H_kwargs,
        ).view(
            self.out_channels,
            H.shape[0],
            self.in_channels // self.groups,
            *self.kernel_size,
        )

        weight = weight if self._mask is None else self._mask * weight
        weight = weight if self.det_H is None else self.det_H * weight

        return weight


class GSeparableKernel(_Kernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        grid_H: Tensor,
        grid_Rn: Tensor,
        groups: int = 1,
        mask: Tensor | None = None,
        det_H: Callable | None = None,
        inverse_H: Callable | None = None,
        left_apply_to_H: Callable | None = None,
        left_apply_to_Rn: Callable | None = None,
        interpolate_H: Callable | None = None,
        interpolate_Rn: Callable | None = None,
        interpolate_H_kwargs: dict = {},
        interpolate_Rn_kwargs: dict = {},
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            groups=groups,
            mask=mask,
            det_H=det_H,
            inverse_H=inverse_H,
            left_apply_to_H=left_apply_to_H,
            left_apply_to_Rn=left_apply_to_Rn,
            interpolate_H=interpolate_H,
            interpolate_Rn=interpolate_Rn,
            interpolate_H_kwargs=interpolate_H_kwargs,
            interpolate_Rn_kwargs=interpolate_Rn_kwargs,
        )

        self.group_kernel_size = grid_H.shape[0]

        self.register_buffer("_grid_H", grid_H)
        self.register_buffer("_grid_Rn", grid_Rn)

        self.weight_H = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, self.group_kernel_size)
        )

        self.weight_Rn = nn.Parameter(torch.empty(out_channels, 1, kernel_size))

        init.kaiming_uniform_(self.weight_H, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_Rn, a=math.sqrt(5))

    def forward(self, in_H: Tensor, out_H: Tensor) -> tuple[Tensor, Tensor]:
        num_in_H, num_out_H = in_H.shape[0], out_H.shape[0]

        out_H_inverse = self.inverse_H(out_H)

        H_product_H = self.left_apply_to_H(out_H_inverse, in_H)
        H_product_Rn = self.left_apply_to_Rn(out_H_inverse, self._grid_Rn)

        # interpolate SO3
        weight_H = (
            self.interpolate_H(
                H_product_H.view(-1, 3, 3),
                self.weight.transpose(0, 2).reshape(1, self.num_H, -1),
                **self.interpolate_H_kwargs,
            )
            .view(
                num_in_H,
                num_out_H,
                self.in_channels // self.groups,
                self.out_channels,
                *self.kernel_size,
            )
            .transpose(0, 3)
            .transpose(1, 3)
        )

        # interpolate R3
        weight_Rn = self.interpolate_Rn(
            self.weight.repeat_interleave(num_out_H, dim=0),
            H_product_Rn.repeat(self.out_channels, 1, 1, 1, 1),
            **self.interpolate_H_kwargs,
        ).view(
            self.out_channels,
            num_out_H,
            self.in_channels // self.groups,
            *self.kernel_size,
        )

        weight_Rn = weight_Rn if self.mask is None else self.mask * weight_Rn
        weight_Rn = weight_Rn if self.det_H is None else self.det_H * weight_Rn

        return weight_H, weight_Rn


class GSubgroupKernel(_Kernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        grid_H: Tensor,
        groups: int = 1,
        det_H: Callable | None = None,
        inverse_H: Callable | None = None,
        left_apply_to_H: Callable | None = None,
        interpolate_H: Callable | None = None,
        interpolate_H_kwargs: dict = {},
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            groups=groups,
            det_H=det_H,
            inverse_H=inverse_H,
            left_apply_to_H=left_apply_to_H,
            interpolate_H=interpolate_H,
            interpolate_H_kwargs=interpolate_H_kwargs,
        )

        self.group_kernel_size = grid_H.shape[0]

        self.register_buffer("_grid_H", grid_H)

        self.weight_H = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, self.group_kernel_size)
        )

        init.kaiming_uniform_(self.weight_H, a=math.sqrt(5))

    def forward(self, in_H: Tensor, out_H: Tensor) -> tuple[Tensor, Tensor]:
        num_in_H, num_out_H = in_H.shape[0], out_H.shape[0]

        out_H_inverse = self.inverse_H(out_H)

        H_product_H = self.left_apply_to_H(out_H_inverse, in_H)

        # interpolate SO3
        weight = (
            self.interpolate_H(
                H_product_H.view(-1, 3, 3),
                self.weight.transpose(0, 2).reshape(1, self.num_H, -1),
                **self.interpolate_H_kwargs,
            )
            .view(
                num_in_H,
                num_out_H,
                self.in_channels // self.groups,
                self.out_channels,
                *self.kernel_size,
            )
            .transpose(0, 3)
            .transpose(1, 3)
        )

        return weight


class GKernel(_Kernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        grid_H: Tensor,
        grid_Rn: Tensor,
        groups: int = 1,
        mask: Tensor | None = None,
        det_H: Callable | None = None,
        inverse_H: Callable | None = None,
        left_apply_to_H: Callable | None = None,
        left_apply_to_Rn: Callable | None = None,
        interpolate_H: Callable | None = None,
        interpolate_Rn: Callable | None = None,
        interpolate_H_kwargs: dict = {},
        interpolate_Rn_kwargs: dict = {},
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            groups=groups,
            mask=mask,
            det_H=det_H,
            inverse_H=inverse_H,
            left_apply_to_H=left_apply_to_H,
            left_apply_to_Rn=left_apply_to_Rn,
            interpolate_H=interpolate_H,
            interpolate_Rn=interpolate_Rn,
            interpolate_H_kwargs=interpolate_H_kwargs,
            interpolate_Rn_kwargs=interpolate_Rn_kwargs,
        )

        self.group_kernel_size = grid_H.shape[0]

        self.register_buffer("_grid_H", grid_H)
        self.register_buffer("_grid_Rn", grid_Rn)

        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                self.group_kernel_size,
                *kernel_size,
            )
        )

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, in_H: Tensor, out_H: Tensor) -> Tensor:
        num_in_H, num_out_H = in_H.shape[0], out_H.shape[0]

        out_H_inverse = self.inverse_H(out_H)

        H_product_H = self.left_apply_to_H(out_H_inverse, in_H)
        H_product_Rn = self.left_apply_to_Rn(out_H_inverse, self._grid_Rn)

        # interpolate SO3
        weight = self.interpolate_H(
            H_product_H.view(-1, 3, 3),
            self._grid_H,
            self.weight.transpose(0, 2).reshape(self.num_H, -1),
            **self.interpolate_H_kwargs,
        ).view(
            num_out_H * num_in_H,
            (self.in_channels // self.groups) * self.out_channels,
            *self.kernel_size,
        )

        # interpolate R3
        weight = (
            self.interpolate_Rn(
                weight,
                H_product_Rn.repeat_interleave(num_in_H, dim=0),
                **self.interpolate_Rn_kwargs,
            )
            .view(
                num_out_H,
                num_in_H,
                self.in_channels // self.groups,
                self.out_channels,
                *self.kernel_size,
            )
            .transpose(0, 3)
            .transpose(1, 3)
        )

        weight = weight if self.mask is None else self.mask * weight
        weight = weight if self.det_H is None else self.det_H * weight

        return weight