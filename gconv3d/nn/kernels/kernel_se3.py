"""
kernel_se3.py

Implements continuous kernels for SE3 convolutions,
separable SE3 convolutions, and lifting SE3 convolutions.
"""
from __future__ import annotations

from typing import Optional

from gconv3d.nn.kernels import GKernel, GSeparableKernel, GLiftingKernel

from torch import Tensor

from gconv3d.geometry import rotation as R
from gconv3d.nn import functional as gF


class GLiftingKernelSE3(GLiftingKernel):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        group_kernel_size: int = 4,
        groups: int = 1,
        mode: str = "bilinear",
        padding_mode: str = "border",
        mask: bool = True,
        grid_H: Optional[Tensor] = None,
    ):

        if grid_H is None:
            grid_H = gF.create_grid_SO3("uniform", group_kernel_size, "matrix")

        grid_Rn = gF.create_grid_R3(kernel_size)

        mask = gF.create_spherical_mask(kernel_size) if mask else None

        interpolate_Rn_kwargs = {"mode": mode, padding_mode: padding_mode}

        super().__init__(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            grid_H,
            grid_Rn,
            groups,
            mask=mask,
            inverse_H=R.matrix_inverse,
            left_apply_to_Rn=R.left_apply_to_R3,
            interpolate_Rn=gF.grid_sample,
            interpolate_Rn_kwargs=interpolate_Rn_kwargs,
        )


class GSeparableKernelSE3(GSeparableKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int,
        groups: int = 1,
        group_mode: str = "rbf",
        group_mode_width: float = 0.0,
        spatial_mode: str = "bilinear",
        spatial_padding_mode: str = "border",
        mask: bool = True,
        grid_H: Optional[Tensor] = None,
    ) -> None:

        if grid_H is None:
            grid_H = gF.create_grid_SO3("uniform", group_kernel_size, "matrix")

        grid_Rn = gF.create_grid_R3(kernel_size)

        width = (
            group_mode_width
            if group_mode_width
            else 0.8 * R.nearest_neighbour_distance(grid_H).mean()
        )

        interpolate_H_kwargs = {"mode": group_mode, "width": width}
        interpolate_Rn_kwargs = {
            "mode": spatial_mode,
            "padding_mode": spatial_padding_mode,
        }

        mask = gF.create_spherical_mask(kernel_size) if mask else None

        super().__init__(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            grid_H,
            grid_Rn,
            groups,
            mask=mask,
            inverse_H=R.matrix_inverse,
            left_apply_to_H=R.left_apply_to_matrix,
            left_apply_to_Rn=R.left_apply_to_R3,
            interpolate_H=gF.so3_sample,
            interpolate_Rn=gF.grid_sample,
            interpolate_H_kwargs=interpolate_H_kwargs,
            interpolate_Rn_kwargs=interpolate_Rn_kwargs,
        )


class GKernelSE3(GKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int,
        groups: int = 1,
        group_mode: str = "rbf",
        group_mode_width: float = 0.0,
        spatial_mode: str = "bilinear",
        spatial_padding_mode: str = "border",
        mask: bool = True,
        grid_H: Optional[Tensor] = None,
    ) -> None:

        if grid_H is None:
            grid_H = gF.create_grid_SO3("uniform", group_kernel_size, "matrix")

        grid_Rn = gF.create_grid_R3(kernel_size)

        width = (
            group_mode_width
            if group_mode_width
            else 0.8 * R.nearest_neighbour_distance(grid_H).mean()
        )

        interpolate_H_kwargs = {"mode": group_mode, "width": width}
        interpolate_Rn_kwargs = {
            "mode": spatial_mode,
            "padding_mode": spatial_padding_mode,
        }

        mask = gF.create_spherical_mask(kernel_size) if mask else None

        super().__init__(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            grid_H,
            grid_Rn,
            groups,
            mask=mask,
            inverse_H=R.matrix_inverse,
            left_apply_to_H=R.left_apply_to_matrix,
            left_apply_to_Rn=R.left_apply_to_R3,
            interpolate_H=gF.so3_sample,
            interpolate_Rn=gF.grid_sample,
            interpolate_H_kwargs=interpolate_H_kwargs,
            interpolate_Rn_kwargs=interpolate_Rn_kwargs,
        )
