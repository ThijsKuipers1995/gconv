"""
kernel.py

Contains base classes for group kernel, separable group kernel, and
lifting kernels.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.init as init

from torch import Tensor

import math


class GroupKernel(nn.Module):
    __constants__ = [
        "in_channels",
        "out_channels",
        "kernel_size",
        "group_kernel_size",
        "groups",
        "mask",
        "grid_H",
        "grid_Rn",
        "det_H",
        "inverse_H",
        "left_apply_to_H",
        "left_apply_to_Rn",
        "interpolate_H",
        "interpolate_Rn",
        "_group_kernel_dim",
        "interpolate_H_kwargs",
        "interpolate_Rn_kwargs",
    ]

    def reset_parameters(self) -> None:
        ...

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        group_kernel_size: tuple,
        groups: int = 1,
        grid_H: Optional[Tensor] = None,
        grid_Rn: Optional[Tensor] = None,
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
        """
        The group kernel manages the group and sampling of weights.

        Arguments:
            - in_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: tuple denoting the spatial kernel size.
            - group_kernel_size: tuple where each element denotes the size of the
                                 corresponding subgroup weight.
            - groups: int denoting the number of groups for depth-wise separability.
            - grid_H: tensor containing the reference grid of group elements.
            - grid_Rn: tensor containing the reference spatial grid.
            - mask: tensor containing a mask to be applied to spatial kernels if provided.
            - det_H: callable that returns the determinant of the given group elements.
            - inverse_H: callable that returns the inverse of the given group elements.
            - left_apply_to_H: callable that implements the pairwise group action of H
                               given two grids of group elements.
            - left_apply_to_Rn: callable that implements the group product of on Rn
                                given a grid of group elements and a grid of Rn vectors.
            - interpolate_H: callable that interpolates the weights for a given grid of
                             group elements, the weights, and the reference grid_H.
            - interpolate_Rn: callable that interpolates the weights for a given grids
                              of Rn reference grids transformed by H.
            - interpolate_H_kwargs: dict containing keyword arguments to be passed to
                                    interpolate_H.
            - interpolate_Rn_kwargs: dict containing keyword arguments to be passed to
                                     interpolate_Rn.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.group_kernel_size = group_kernel_size
        self._group_kernel_dim = sum(group_kernel_size)

        self.groups = groups

        self.register_buffer("grid_H", grid_H)
        self.register_buffer("grid_Rn", grid_Rn)

        self.register_buffer("mask", mask)

        self.det_H = det_H
        self.inverse_H = inverse_H
        self.left_apply_to_H = left_apply_to_H
        self.left_apply_to_Rn = left_apply_to_Rn
        self.interpolate_H = interpolate_H
        self.interpolate_Rn = interpolate_Rn
        self.interpolate_H_kwargs = interpolate_H_kwargs
        self.interpolate_Rn_kwargs = interpolate_Rn_kwargs


class GLiftingKernel(GroupKernel):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        group_kernel_size: tuple,
        grid_H,
        grid_Rn,
        groups: int = 1,
        mask: Tensor | None = None,
        det_H: Callable | None = None,
        inverse_H: Callable | None = None,
        left_apply_to_Rn: Callable | None = None,
        interpolate_Rn: Callable | None = None,
        interpolate_Rn_kwargs: dict = {},
    ) -> None:
        """
        The lifting kernel manages the group and weights for
        lifting an Rn input to Rn x H input.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size,
            groups=groups,
            grid_H=grid_H,
            grid_Rn=grid_Rn,
            mask=mask,
            det_H=det_H,
            inverse_H=inverse_H,
            left_apply_to_Rn=left_apply_to_Rn,
            interpolate_Rn=interpolate_Rn,
            interpolate_Rn_kwargs=interpolate_Rn_kwargs,
        )

        self.weight = torch.nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )

        # for expanding determinant to correct size
        self.weight_dims = (1,) * (self.weight.ndim - 2)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, H) -> Tensor:
        num_H = H.shape[0]

        H_product = self.left_apply_to_Rn(self.inverse_H(H), self.grid_Rn)

        product_dims = (1,) * (H_product.ndim - 1)

        weight = self.interpolate_Rn(
            self.weight.repeat_interleave(H.shape[0], dim=0),
            H_product.repeat(self.out_channels, *product_dims),
            **self.interpolate_H_kwargs,
        ).view(
            self.out_channels, num_H, self.in_channels // self.groups, *self.kernel_size
        )

        if self.mask is not None:
            weight = self.mask * weight

        if self.det_H is not None:
            weight = weight / self.det_H(H).view(-1, *self.weight_dims)

        return weight


class GSeparableKernel(GroupKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        group_kernel_size: tuple,
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
        """
        The separable kernel manages the group and weights for
        separable group convolutions, returning weights for
        subgroup H and Rn separately.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size=group_kernel_size,
            grid_H=grid_H,
            grid_Rn=grid_Rn,
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

        self.weight_H = nn.Parameter(
            torch.empty(self._group_kernel_dim, out_channels, in_channels // groups)
        )

        self.weight = nn.Parameter(torch.empty(out_channels, 1, *kernel_size))

        # for expanding determinant to correct size
        self.weight_dims = (1,) * len(kernel_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_H, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, in_H: Tensor, out_H: Tensor) -> tuple[Tensor, Tensor]:
        num_in_H, num_out_H = in_H.shape[0], out_H.shape[0]
        H_dims = in_H.shape[1:]

        out_H_inverse = self.inverse_H(out_H)

        H_product_H = self.left_apply_to_H(out_H_inverse, in_H)
        H_product_Rn = self.left_apply_to_Rn(out_H_inverse, self.grid_Rn)

        product_dims = (1,) * H_product_H.ndim

        # interpolate SO3
        weight_H = (
            self.interpolate_H(
                H_product_H.view(-1, *H_dims),
                self.weight_H.reshape(self._group_kernel_dim, -1),
                self.grid_H,
                **self.interpolate_H_kwargs,
            )
            .view(
                num_in_H,
                num_out_H,
                self.in_channels // self.groups,
                self.out_channels,
                *self.weight_dims,
            )
            .transpose(0, 3)
            .transpose(1, 3)
        )

        # interpolate R3
        weight = self.interpolate_Rn(
            self.weight.repeat_interleave(num_out_H, dim=0),
            H_product_Rn.repeat(self.out_channels, *product_dims),
            **self.interpolate_Rn_kwargs,
        ).view(
            self.out_channels,
            num_out_H,
            1,
            *self.kernel_size,
        )

        if self.mask is not None:
            weight = self.mask * weight

        if self.det_H is not None:
            weight = weight / self.det_H(out_H).view(-1, *self.weight_dims)

        return weight_H, weight


class GSubgroupKernel(GroupKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        group_kernel_size: tuple,
        grid_H: Tensor,
        groups: int = 1,
        det_H: Callable | None = None,
        inverse_H: Callable | None = None,
        left_apply_to_H: Callable | None = None,
        interpolate_H: Callable | None = None,
        interpolate_H_kwargs: dict = {},
    ) -> None:
        """
        The subgroup kernel manages the group and weights for
        convolutions applied only to the subgroup H.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size=group_kernel_size,
            grid_H=grid_H,
            groups=groups,
            det_H=det_H,
            inverse_H=inverse_H,
            left_apply_to_H=left_apply_to_H,
            interpolate_H=interpolate_H,
            interpolate_H_kwargs=interpolate_H_kwargs,
        )

        self.weight = nn.Parameter(
            torch.empty(self._group_kernel_dim, out_channels, in_channels // groups)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, in_H: Tensor, out_H: Tensor) -> Tensor:
        num_in_H, num_out_H = in_H.shape[0], out_H.shape[0]
        H_dims = in_H.shape[1:]

        out_H_inverse = self.inverse_H(out_H)

        H_product_H = self.left_apply_to_H(out_H_inverse, in_H)

        # interpolate SO3
        weight = (
            self.interpolate_H(
                H_product_H.view(-1, *H_dims),
                self.weight.reshape(self._group_kernel_dim, -1),
                self.grid_H,
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


class GKernel(GroupKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        group_kernel_size: tuple,
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
        """
        Manages the group and weights for the full group convolution.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size=group_kernel_size,
            grid_H=grid_H,
            grid_Rn=grid_Rn,
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

        self.weight = nn.Parameter(
            torch.empty(
                self._group_kernel_dim,
                out_channels,
                in_channels // groups,
                *kernel_size,
            )
        )

        # for expanding determinant to correct size
        self.weight_dims = (1,) * (self.weight.ndim - 2)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, in_H: Tensor, out_H: Tensor) -> Tensor:
        num_in_H, num_out_H = in_H.shape[0], out_H.shape[0]
        H_dims = in_H.shape[1:]

        out_H_inverse = self.inverse_H(out_H)

        H_product_H = self.left_apply_to_H(out_H_inverse, in_H)
        H_product_Rn = self.left_apply_to_Rn(out_H_inverse, self.grid_Rn)

        # interpolate SO3
        weight = self.interpolate_H(
            H_product_H.view(-1, *H_dims),
            self.weight.reshape(self._group_kernel_dim, -1),
            self.grid_H,
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

        if self.mask is not None:
            weight = self.mask * weight

        if self.det_H is not None:
            weight = weight / self.det_H(out_H).view(-1, *self.weight_dims)

        return weight
