from __future__ import annotations

from typing import Optional

from gconv.nn.kernels import GLiftingKernel, GSeparableKernel, GKernel

from torch import Tensor

from gconv.geometry import o3, so3
from gconv.nn import functional as gF


class GLiftingKernelE3(GLiftingKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: tuple | int = (4, 4),
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
            - group_kernel_size: int or tuple denoting the kernel size for (rotations, reflections).
                                 If provided as int, given size will be used for both rotations and
                                 reflections, i.e., the total kernel will be double.
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
        if isinstance(group_kernel_size, int):
            kernel_size = (group_kernel_size, group_kernel_size)

        if grid_H is None:
            grid_H = o3.uniform_grid(group_kernel_size, "matrix")

        grid_Rn = gF.create_grid_R3(kernel_size)

        mask = gF.create_spherical_mask_R3(kernel_size) if mask else None

        sample_Rn_kwargs = {
            "mode": sampling_mode,
            "padding_mode": sampling_padding_mode,
        }
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
            sample_Rn_kwargs,
        )


class GSeparableKernelE3(GSeparableKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: tuple[int, int],
        groups: int = 1,
        group_sampling_mode: str = "rbf",
        group_rotation_sampling_width: float = 0.0,
        group_reflection_sampling_width: float = 0.0,
        spatial_sampling_mode: str = "bilinear",
        spatial_sampling_padding_mode: str = "border",
        mask: bool = True,
        grid_H: Optional[Tensor] = None,
    ) -> None:
        """
        Implements SE3 separable kernel.

        Arguments:
            - in_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: int denoting the spatial kernel size.
            - group_kernel_size: int or tuple denoting the kernel size for (rotations, reflections).
                                 If provided as int, given size will be used for both rotations and
                                 reflections, i.e., the total kernel will be double.
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
            grid_H = o3.uniform_grid(sum(group_kernel_size), True)

        grid_Rn = gF.create_grid_R3(kernel_size)

        if not rotation_sampling_width:
            rotation_sampling_width = (
                0.8
                * so3.nearest_neighbour_distance(grid_H[: group_kernel_size[0]]).mean()
            )

        if not reflection_sampling_width:
            reflection_sampling_width = (
                0.8
                * so3.nearest_neighbour_distance(grid_H[group_kernel_size[0] :]).mean()
            )

        sample_H_kwargs = {
            "signal_grid_size": group_kernel_size,
            "mode": group_sampling_mode,
            "rotation_width": group_rotation_sampling_width,
            "reflection_width": group_reflection_sampling_width,
        }
        sample_Rn_kwargs = {
            "mode": spatial_sampling_mode,
            "padding_mode": spatial_sampling_padding_mode,
        }

        mask = gF.create_spherical_mask_R3(kernel_size) if mask else None

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
            sample_H_kwargs,
            sample_Rn_kwargs,
        )


class GKernelE3(GKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: tuple[int, int],
        groups: int = 1,
        group_sampling_mode: str = "rbf",
        group_rotation_sampling_width: float = 0.0,
        group_reflection_sampling_width: float = 0.0,
        spatial_sampling_mode: str = "bilinear",
        spatial_sampling_padding_mode: str = "border",
        mask: bool = True,
        grid_H: Optional[Tensor] = None,
    ) -> None:
        """
        Implements SE3 kernel.

        Arguments:
            - in_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: int denoting the spatial kernel size.
            - group_kernel_size: int or tuple denoting the kernel size for (rotations, reflections).
                                 If provided as int, given size will be used for both rotations and
                                 reflections, i.e., the total kernel will be double.
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
            grid_H = o3.uniform_grid(sum(group_kernel_size), True)

        grid_Rn = gF.create_grid_R3(kernel_size)

        if not rotation_sampling_width:
            rotation_sampling_width = (
                0.8
                * so3.nearest_neighbour_distance(grid_H[: group_kernel_size[0]]).mean()
            )

        if not reflection_sampling_width:
            reflection_sampling_width = (
                0.8
                * so3.nearest_neighbour_distance(grid_H[group_kernel_size[0] :]).mean()
            )

        sample_H_kwargs = {
            "signal_grid_size": group_kernel_size,
            "mode": group_sampling_mode,
            "rotation_width": group_rotation_sampling_width,
            "reflection_width": group_reflection_sampling_width,
        }
        sample_Rn_kwargs = {
            "mode": spatial_sampling_mode,
            "padding_mode": spatial_sampling_padding_mode,
        }

        mask = gF.create_spherical_mask_R3(kernel_size) if mask else None

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
            sample_H_kwargs,
            sample_Rn_kwargs,
        )
