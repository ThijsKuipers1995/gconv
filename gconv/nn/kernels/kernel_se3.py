"""
kernel_se3.py

Implements continuous kernels for SE3 convolutions,
separable SE3 convolutions, and lifting SE3 convolutions.
"""
from __future__ import annotations

from typing import Optional

from gconv.nn.kernels import GKernel, GSeparableKernel, GLiftingKernel

from torch import Tensor

from gconv.geometry import so3
from gconv.nn import functional as gF


class GLiftingKernelSE3(GLiftingKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int = 4,
        groups: int = 1,
        sampling_mode: str = "bilinear",
        sampling_padding_mode: str = "border",
        mask: bool = True,
        grid_H: Optional[Tensor] = None,
    ):
        """
        Implements SE3 lifting kernel.

        Arguments:
            - in_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: int denoting the spatial kernel size.
            - group_kernel_size: int denoting the group kernel size.
            - groups: number of groups for depth-wise separability.
            - sampling_mode: str indicating the sampling mode. Supports bilinear (default)
                             or nearest.
            - sampling_padding_mode: str indicating padding mode for sampling. Default
                                     border.
            - mask: bool if true, will initialize spherical mask.
            - grid_H: tensor of reference grid used for interpolation. If not
                      provided, a uniform grid of group_kernel_size will be
                      generated. If provided, will overwrite given group_kernel_size.
        """
        if grid_H is None:
            grid_H = so3.uniform_grid(group_kernel_size, "matrix")

        grid_Rn = gF.create_grid_R3(kernel_size)

        mask = gF.create_spherical_mask(kernel_size) if mask else None

        sample_Rn_kwargs = {
            "mode": sampling_mode,
            "padding_mode": sampling_padding_mode,
        }

        super().__init__(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            (grid_H.shape[0],),
            grid_H,
            grid_Rn,
            groups,
            mask=mask,
            inverse_H=so3.matrix_inverse,
            left_apply_to_Rn=so3.left_apply_to_R3,
            sample_Rn=gF.grid_sample,
            sample_Rn_kwargs=sample_Rn_kwargs,
        )


class GSeparableKernelSE3(GSeparableKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int,
        groups: int = 1,
        group_sampling_mode: str = "rbf",
        group_sampling_width: float = 0.0,
        spatial_sampling_mode: str = "bilinear",
        spatial_sampling_padding_mode: str = "border",
        mask: bool = True,
        grid_H: Optional[Tensor] = None,
    ) -> None:
        """
        Implements SE3 lifting kernel.

        Arguments:
            - in_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: int denoting the spatial kernel size.
            - group_kernel_size: int denoting the group kernel size.
            - groups: number of groups for depth-wise separability.
            - group_sampling_mode: str indicating the sampling mode. Supports rbf (default)
                                   or nearest.
            - group_sampling_width: float denoting the width of the Gaussian rbf kernels.
                                    If 0.0 (default, recommended), width will be initialized
                                    based on grid_H density.
            - spatial_sampling_mode: str indicating the sampling mode. Supports bilinear (default)
                                         or nearest.
            - spatial_sampling_padding_mode: str indicating padding mode for sampling. Default
                                             border.
            - mask: bool if true, will initialize spherical mask.
            - grid_H: tensor of reference grid used for interpolation. If not
                      provided, a uniform grid of group_kernel_size will be
                      generated. If provided, will overwrite given group_kernel_size.
        """
        if grid_H is None:
            grid_H = so3.uniform_grid(group_kernel_size, "matrix")

        grid_Rn = gF.create_grid_R3(kernel_size)

        if not group_sampling_width:
            group_sampling_width = 0.8 * so3.nearest_neighbour_distance(grid_H).mean()

        sample_H_kwargs = {
            "mode": group_sampling_mode,
            "width": group_sampling_width,
        }
        sample_Rn_kwargs = {
            "mode": spatial_sampling_mode,
            "padding_mode": spatial_sampling_padding_mode,
        }

        mask = gF.create_spherical_mask(kernel_size) if mask else None

        super().__init__(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            (grid_H.shape[0],),
            grid_H,
            grid_Rn,
            groups,
            mask=mask,
            inverse_H=so3.matrix_inverse,
            left_apply_to_H=so3.left_apply_to_matrix,
            left_apply_to_Rn=so3.left_apply_to_R3,
            sample_H=so3.grid_sample,
            sample_Rn=gF.grid_sample,
            sample_H_kwargs=sample_H_kwargs,
            sample_Rn_kwargs=sample_Rn_kwargs,
        )


class GKernelSE3(GKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int,
        groups: int = 1,
        group_sampling_mode: str = "rbf",
        group_sampling_width: float = 0.0,
        spatial_sampling_mode: str = "bilinear",
        spatial_sampling_padding_mode: str = "border",
        mask: bool = True,
        grid_H: Optional[Tensor] = None,
    ) -> None:
        """
        Implements SE3 lifting kernel.

        Arguments:
            - in_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: int denoting the spatial kernel size.
            - group_kernel_size: int denoting the group kernel size.
            - groups: number of groups for depth-wise separability.
            - group_sampling_mode: str indicating the sampling mode. Supports rbf (default)
                                   or nearest.
            - group_sampling_width: float denoting the width of the Gaussian rbf kernels.
                                    If 0.0 (default, recommended), width will be initialized
                                    based on grid_H density.
            - spatial_sampling_mode: str indicating the sampling mode. Supports bilinear (default)
                                         or nearest.
            - spatial_sampling_padding_mode: str indicating padding mode for sampling. Default
                                             border.
            - mask: bool if true, will initialize spherical mask.
            - grid_H: tensor of reference grid used for interpolation. If not
                      provided, a uniform grid of group_kernel_size will be
                      generated. If provided, will overwrite given group_kernel_size.
        """
        if grid_H is None:
            grid_H = so3.uniform_grid(group_kernel_size, "matrix")

        grid_Rn = gF.create_grid_R3(kernel_size)

        if not group_sampling_width:
            group_sampling_width = 0.8 * so3.nearest_neighbour_distance(grid_H).mean()

        sample_H_kwargs = {
            "mode": group_sampling_mode,
            "width": group_sampling_width,
        }
        sample_Rn_kwargs = {
            "mode": spatial_sampling_mode,
            "padding_mode": spatial_sampling_padding_mode,
        }

        mask = gF.create_spherical_mask(kernel_size) if mask else None

        super().__init__(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            (grid_H.shape[0],),
            grid_H,
            grid_Rn,
            groups,
            mask=mask,
            inverse_H=so3.matrix_inverse,
            left_apply_to_H=so3.left_apply_to_matrix,
            left_apply_to_Rn=so3.left_apply_to_R3,
            sample_H=so3.grid_sample,
            sample_Rn=gF.grid_sample,
            sample_H_kwargs=sample_H_kwargs,
            sample_Rn_kwargs=sample_Rn_kwargs,
        )
