from typing import Any
from pyparsing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init

from torch import Tensor

import math

from gconv.geometry import so3
from gconv.nn import functional as gF


class _Kernel(nn.Module):
    def _inverse_group_action(self, H: Tensor, X: Tensor) -> Tensor:
        """
        Applies the inverse group action of each element in H to
        each element in X.

        Arguments:
            - H: Tensor of group elements of shape `(N, ...)`.
            - X: Tensor of group elements of shape `(M, ...)`.

        Returns:
            - Tensor containing inverse group actions of H on X
              of shape `(N, M, ...)`.
        """
        raise NotImplementedError

    def _sample(self, H: Tensor, grid: Tensor, weight: Tensor) -> Tensor:
        """
        Samples kernel and returns the weights associated with the
        given group elements `H`. By default, the tensor of group elements,
        a discrete reference grid of group elements and the weights corresponding
        to this grid are passed to this function.

        Arguments:
            - H: Tensor of group elements of shape `(N, ...)`.
            - grid: interpolation grid of group elements of shape `(M, ...)`.
            - weight: Weight tensor defined on `grid` of shape `(Cin, Cout, M, *kernel_size)`.

        Optional Arguments:
            - Any optional keyword arguments can be provided when defining `sample(...)`.
              These will be passed along through the `sample_kwargs` argument provided
              to the class constructor.

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
        group_sample_resolution: Optional[int] = None,
        sample_kwargs: dict[Any] = {},
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
            - group_sample_resolution: Integer denoting the sampling resolution
                                       of the group grid.
            - sample_kwargs: Dictionary containing any keyword arguments that
                             are passed on to `_sample(...)`.

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

        self.group_sample_resolution = group_sample_resolution

        self.sample_kwargs = sample_kwargs

        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                self.group_resolution,
                *self.kernel_size,
            )
        )

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, H: Tensor, X: Optional[Tensor] = None) -> Tensor:
        """
        Applies the inverse group action of H on X and interpolates
        weight.

        Arguments:
            - H: Tensor of group elements of shape `(N, ...)`.
            - X: Tensor of group elements of shape `(M, ...)`. If not provided,
                 a grid will be sampled according to `_sample_grid`.

        Returns:
            - weight corresponding to transformed X.
        """
        X = X if X is not None else self._sample_group(self.group_sample_resolution)
        X_t = self._inverse_group_action(H, X)
        weight = self._sample(X_t, self.grid, self.weight, **self.sample_kwargs)
        return weight

    def extra_repr(self) -> str:
        s = ""
        s += f"in_channels={self.in_channels}"
        s += f"out_channels={self.out_channels}"
        s += f"kernel_size={self.kernel_size}"
        if self.groups > 1:
            s += f", groups={self.groups}"
        s += f", group_resolution={self.group_resolution}"
        if self.group_resolution:
            s += f"group_sample_resolution={self.group_sample_resolution}"
        s += f", sample_kwargs={self.sample_kwargs}"
        return s


class KernelSO3(_Kernel):
    so3_interpolation_modes = ["nearest", "rbf"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group_resolution: int,
        groups: int = 1,
        grid: Optional[Tensor] = None,
        mode: str = "rbf",
        group_sample_resolution: Optional[int] = None,
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
            group_sample_resolution=group_sample_resolution,
        )

    def _inverse_group_action(self, H: Tensor, X: Tensor) -> Tensor:
        return so3.left_apply_to_matrix(so3.matrix_inverse(H)[:, None], X)

    def _sample(
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


class KernelSO3R3(_Kernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        groups: int = 1,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        group_sample_resolution: Optional[int] = None,
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

        sample_kwargs = {"mode": mode, "padding_mode": padding_mode}

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            groups,
            grid,
            sample_kwargs,
            group_sample_resolution=group_sample_resolution,
        )

    def _inverse_group_action(self, H: Tensor, X: Tensor) -> Tensor:
        return so3.left_apply_to_R3(so3.matrix_inverse(H), X)

    def _sample(
        self,
        _,
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
