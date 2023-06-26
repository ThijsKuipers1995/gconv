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
        """
        Implements SE2 lifting kernel.

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
            left_apply_to_Rn=so2.left_apply_angle_to_R2,
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
            left_apply_to_H=so2.left_apply_to_angle,
            left_apply_to_Rn=so2.left_apply_angle_to_R2,
            sample_H=so2.grid_sample,
            sample_Rn=gF.grid_sample,
            sample_H_kwargs=sample_H_kwargs,
            sample_Rn_kwargs=sample_Rn_kwargs,
        )


class GKernelSE2(GKernel):
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
            left_apply_to_H=so2.left_apply_to_angle,
            left_apply_to_Rn=so2.left_apply_angle_to_R2,
            sample_H=so2.grid_sample,
            sample_Rn=gF.grid_sample,
            sample_H_kwargs=sample_H_kwargs,
            sample_Rn_kwargs=sample_Rn_kwargs,
        )
