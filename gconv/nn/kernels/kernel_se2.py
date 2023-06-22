from gconv.nn.kernels import GLiftingKernel, GSeparableKernel, GKernel
from gconv.geometry import so2

import gconv.nn.functional as gF

from torch import Tensor


class GLiftingKernelSE2(GLiftingKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int = 8,
        groups: int = 1,
        sampling_mode: str = "bilinear",
        sampling_padding_mode: str = "border",
        mask: bool = True,
        grid_H: Tensor | None = None,
    ) -> None:

        if grid_H is None:
            grid_H = so2.uniform_grid(group_kernel_size)

        grid_Rn = gF.create_grid_R2(kernel_size)

        mask = gF.create_spherical_mask_R2(kernel_size)

        sample_Rn_kwargs = {
            "mode": sampling_mode,
            "padding_mode": sampling_padding_mode,
        }

        super().__init__(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            (grid_H.shape[0],),
            grid_H,
            grid_Rn,
            groups,
            mask=mask,
            inverse_H=so2.inverse_angle,
            left_apply_to_Rn=so2.left_apply_to_angle,
            sample_Rn=gF.grid_sample,
            sample_Rn_kwargs=sample_Rn_kwargs,
        )


class GSeparableKernelSE2(GSeparableKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int = 8,
        groups: int = 1,
        group_sampling_mode: str = "rbf",
        group_sampling_width: float = 0.0,
        spatial_sampling_mode: str = "bilinear",
        spatial_sampling_padding_mode: str = "border",
        mask: bool = True,
        grid_H: Tensor | None = None,
    ) -> None:

        if grid_H is None:
            grid_H = so2.uniform_grid(group_kernel_size)

        grid_Rn = gF.create_grid_R2(kernel_size)

        if not group_sampling_width:
            group_sampling_width = 0.8 * so2.nearest_neighbour_distance(grid_H).mean()

        sample_H_kwargs = {"mode": group_sampling_mode, "width": group_sampling_width}
        sample_Rn_kwargs = {
            "mode": spatial_sampling_mode,
            "padding_mode": spatial_sampling_padding_mode,
        }

        mask = gF.create_spherical_mask_R2(kernel_size)

        super().__init__(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            (grid_H.shape[0],),
            grid_H,
            grid_Rn,
            groups,
            mask=mask,
            inverse_H=so2.inverse_angle,
            left_apply_to_Rn=so2.left_apply_to_angle,
            sample_H=so2.grid_sample,
            sample_Rn=gF.grid_sample,
            sample_H_kwargs=sample_H_kwargs,
            sample_Rn_kwargs=sample_Rn_kwargs,
        )


class GSeparableKernelSE2(GKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int = 8,
        groups: int = 1,
        group_sampling_mode: str = "rbf",
        group_sampling_width: float = 0.0,
        spatial_sampling_mode: str = "bilinear",
        spatial_sampling_padding_mode: str = "border",
        mask: bool = True,
        grid_H: Tensor | None = None,
    ) -> None:

        if grid_H is None:
            grid_H = so2.uniform_grid(group_kernel_size)

        grid_Rn = gF.create_grid_R2(kernel_size)

        if not group_sampling_width:
            group_sampling_width = 0.8 * so2.nearest_neighbour_distance(grid_H).mean()

        sample_H_kwargs = {"mode": group_sampling_mode, "width": group_sampling_width}
        sample_Rn_kwargs = {
            "mode": spatial_sampling_mode,
            "padding_mode": spatial_sampling_padding_mode,
        }

        mask = gF.create_spherical_mask_R2(kernel_size)

        super().__init__(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            (grid_H.shape[0],),
            grid_H,
            grid_Rn,
            groups,
            mask=mask,
            inverse_H=so2.inverse_angle,
            left_apply_to_Rn=so2.left_apply_to_angle,
            sample_H=so2.grid_sample,
            sample_Rn=gF.grid_sample,
            sample_H_kwargs=sample_H_kwargs,
            sample_Rn_kwargs=sample_Rn_kwargs,
        )
