from typing import Any
from pyparsing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init

from torch import Tensor

import math

from gconv.geometry import so3
from gconv.nn import functional as gF


class _GKernel(nn.Module):
    def inverse_group_action(self, H: Tensor, X: Tensor) -> Tensor:
        """
        Applies the inverse group action of each element in H to
        each element in X.

        Arguments:
            - H: Tensor of group elements.
            - X: Tensor of group elements.

        Returns:
            - Tensor containing inverse group actions of H on X.
        """
        raise NotImplementedError

    def sample(
        self, H: Tensor, grid: Tensor, weight: Tensor, **kwargs: dict[Any]
    ) -> Tensor:
        """
        Returns interpolated signal corresponding to group elements in H based
        on reference interpolation grid and weight defined on the reference grid.

        Arguments:
            - H: Tensor of group elements.
            - interpolation_grid: interpolation grid of group elements.
            - weight: Signal defined on reference grid.

        Returns:
            - Tensor containing interpolated weight.
        """
        raise NotImplementedError

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        groups: int = 1,
        grid: Optional[Tensor] = None,
        interpolation_kwargs: dict[Any] = {},
    ) -> None:
        """
        Base class for interpolation group kernels.

        Arguments:
            - in_channels: Number of input channels.
            - out_channels: Number of output channels.
            - kernel_size: Int or tuple defining kernel size.
            - groups: Number of groups that have separate weights (channel-wise),
                      see Pytorch ConvNd documentation for details.
            - grid: Tensor of reference grid of group elements that will be acted on,
                                  used for interpolation.
            - interpolation_kwargs: Dictionary containing any keyword arguments that
                                    are passed on to `interpolate(...)`.

        Requires:
            - Implementation of the `inverse_group_action` and `interpolate` methods for
              a given group.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups

        self.group_resolution = grid[0]
        self.register_buffer("grid", grid)

        self.interpolation_kwargs = interpolation_kwargs

        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                self.group_resolution,
                *self.kernel_size,
            )
        )

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, H: Tensor, X: Tensor) -> Tensor:
        """
        Applies the inverse group action of H on X and interpolates
        weight.

        Arguments:
            - H: Tensor of group elements.
            - X: Tensor of group elements.

        Returns:
            - weight corresponding to transformed X.
        """
        X_transformed = self.inverse_group_action(H, X)
        weight = self.sample(
            X_transformed,
            self.grid,
            self.weight,
            **self.interpolation_kwargs,
        )
        return weight

    def extra_repr(self) -> str:
        s = ""
        s += f"{self.kernel_size=}"
        if self.groups > 1:
            s += f", {self.groups=}"
        s += f", group_resolution={self.group_resolution}"
        s += f", {self.interpolation_kwargs=}"
        return s


class GKernelSO3onSO3(_GKernel):
    so3_interpolation_modes = ["nearest", "rbf"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group_resolution: int,
        groups: int = 1,
        grid: Optional[Tensor] = None,
        mode: str = "rbf",
    ) -> None:
        """
        Kernel that samples weights defined on a discrete SO3
        grid after SO3 inverse group action.

        Arguments:
            - in_channels: Number of input channels.
            - out_channels: Number of output channels.
            - group_resolution: Resolution of the group kernel.
            - groups: Number of groups that have separate weights (channel-wise),
                      see Pytorch ConvNd documentation for details.
            - grid: Tensor of reference grid of group elements that will be acted on,
                    used for interpolation. If not provided, will initialize a
                    uniform SO3 grid with `group_resolution`.
            - mode: Interpolation mode, either `nearest` or `rbf
        """
        if mode not in self.so3_interpolation_modes:
            raise ValueError(
                f"`mode` must be in {self.so3_interpolation_modes}, but is {mode=}."
            )

        grid = (
            grid
            if grid is not None
            else gF.create_grid_SO3("uniform", group_resolution)
        )

        interpolation_kwargs = {
            "mode": mode,
            "width": so3.nearest_neighbour_distance(grid).mean().item(),
        }

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=(1,),
            groups=groups,
            grid=grid,
            interpolation_kwargs=interpolation_kwargs,
        )

    def inverse_group_action(self, H: Tensor, X: Tensor) -> Tensor:
        return so3.left_apply_to_matrix(so3.matrix_inverse(H)[:, None], X)

    def sample(
        self, H: Tensor, grid: Tensor, weight: Tensor, width: float = 1
    ) -> Tensor:
        _weight = (
            gF.so3_sample(
                weight.transpose(0, 2).reshape(1, self.grid_resolution, -1),
                grid[None],
                so3.matrix_to_quat(H).view(1, -1, 3, 3),
                width=width,
            )
            .view(
                H.shape[0],
                H.shape[1],
                weight.shape[1],
                weight.shape[0],
                weight.shape[-1],
            )
            .transpose(0, 3)
            .transpose(1, 3)
        )

        return _weight


class GKernelSO3onR3(_GKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        groups: int = 1,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
    ) -> None:
        """
        Kernel that samples spatial weights, i.e., R3 after
        applying the SO3 inverse group action.

        Arguments:
            - in_channels: Number of input channels.
            - out_channels: Number of output channels.
            - kernel_size: Int or tuple defining kernel size.
            - groups: Number of groups that have separate weights (channel-wise).
                      See Pytorch ConvNd documentation for details.
            - mode: Interpolation mode, either `nearest` or `bilinear`.
            - padding_mode: Defaults to zeros.
        """
        kernel_size = kernel_size if type(kernel_size) is tuple else 3 * (kernel_size,)
        grid = gF.create_grid_R3(kernel_size[0])[None]

        interpolation_kwargs = {"mode": mode, "padding_mode": padding_mode}

        super().__init__(
            in_channels, out_channels, kernel_size, groups, grid, interpolation_kwargs
        )

    def inverse_group_action(self, H: Tensor, X: Tensor) -> Tensor:
        return so3.left_apply_to_R3(so3.matrix_inverse(H), X)

    def sample(
        self,
        _: Tensor,
        grid: Tensor,
        weight: Tensor,
        mode: str = "bilinear",
        padding_mode="zeros",
    ) -> Tensor:
        _weight = gF.grid_sample(
            weight.squeeze(2),
            grid,
            mode=mode,
            padding_mode=padding_mode,
        ).view(weight.shape[0], grid.shape[0], weight.shape[1], weight.shape[2:])

        return _weight

    def forward(self, H: Tensor) -> Tensor:
        return super().forward(H, self.grid[0])
