from __future__ import annotations

from typing import Callable, Optional

from gconv3d.nn.kernels import GLiftingKernel, GSeparableKernel

from torch import Tensor

from gconv3d.geometry import o3, so3
from gconv3d.nn import functional as gF


class GLiftingKernelE3(GLiftingKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: tuple | int = (4, 4),
        groups: int = 1,
        mode: str = "bilinear",
        padding_mode: str = "border",
        mask: bool = True,
        grid_H: Optional[Tensor] = None,
    ):
        if isinstance(group_kernel_size, int):
            kernel_size = (group_kernel_size, group_kernel_size)

        if grid_H is None:
            grid_H = o3.uniform_grid(group_kernel_size, "matrix")

        grid_Rn = gF.create_grid_R3(kernel_size)

        mask = gF.create_spherical_mask(kernel_size) if mask else None

        interpolate_Rn_kwargs = {"mode": mode, padding_mode: padding_mode}
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size,
            grid_H,
            grid_Rn,
            groups,
            mask,
            o3.det,
            o3.inverse,
            o3.left_apply_to_R3,
            gF.grid_sample,
            interpolate_Rn_kwargs,
        )


class GSeparableKernelE3(GSeparableKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: tuple[int, int],
        groups: int = 1,
        group_mode: str = "rbf",
        rotation_width: float = 0.0,
        reflection_width: float = 0.0,
        spatial_mode: str = "bilinear",
        spatial_padding_mode: str = "border",
        mask: bool = True,
        grid_H: Optional[Tensor] = None,
    ) -> None:

        if grid_H is None:
            grid_H = o3.uniform_grid(sum(group_kernel_size), True)

        grid_Rn = gF.create_grid_R3(kernel_size)

        if not rotation_width:
            rotation_width = (
                0.8
                * so3.nearest_neighbour_distance(grid_H[: group_kernel_size[0]]).mean()
            )

        if not reflection_width:
            reflection_width = (
                0.8
                * so3.nearest_neighbour_distance(grid_H[group_kernel_size[0] :]).mean()
            )

        interpolate_H_kwargs = {
            "signal_grid_size": group_kernel_size,
            "mode": group_mode,
            "rotation_width": rotation_width,
            "reflection_width": reflection_width,
        }
        interpolate_Rn_kwargs = {
            "mode": spatial_mode,
            "padding_mode": spatial_padding_mode,
        }

        mask = gF.create_spherical_mask(kernel_size) if mask else None

        super().__init__(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            group_kernel_size,
            grid_H,
            grid_Rn,
            groups,
            mask,
            o3.det,
            o3.inverse,
            o3.left_apply_to_O3,
            o3.left_apply_to_R3,
            o3.grid_sample,
            gF.grid_sample,
            interpolate_H_kwargs,
            interpolate_Rn_kwargs,
        )
