"""
kernel_so3.py

Implements a kernel for SO3 convolutions.
"""
from __future__ import annotations

from typing import Optional

from gconv.gnn.kernels import GSubgroupKernel

from torch import Tensor

from gconv.geometry import so3


class GSubgroupKernelSO3(GSubgroupKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group_kernel_size: int,
        groups: int = 1,
        sampling_mode: str = "rbf",
        sampling_width: float = 0.0,
        grid_H: Optional[Tensor] = None,
    ) -> None:
        """
        Implements SO3 kernel.

        Arguments:
            - in_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - group_kernel_size: int denoting the group kernel size.
            - groups: number of groups for depth-wise separability.
            - sampling_mode: str indicating the sampling mode. Supports rbf (default)
                             or nearest.
            - sampling_width: float denoting the width of the Gaussian rbf kernels.
                              If 0.0 (default, recommended), width will be initialized
                              based on grid_H density.
            - grid_H: tensor of reference grid used for interpolation. If not
                      provided, a uniform grid of group_kernel_size will be
                      generated. If provided, will overwrite given group_kernel_size.
        """
        if grid_H is None:
            grid_H = so3.uniform_grid(size=group_kernel_size)

        if not sampling_width:
            sampling_width = 0.8 * so3.nearest_neighbour_distance(grid_H).mean()

        sample_H_kwargs = {"mode": sampling_mode, "width": sampling_width}

        super().__init__(
            in_channels,
            out_channels,
            (1, 1, 1),
            (grid_H.shape[0],),
            grid_H,
            groups,
            inverse_H=so3.matrix_inverse,
            left_apply_to_H=so3.left_apply_to_matrix,
            sample_H=so3.grid_sample,
            sample_H_kwargs=sample_H_kwargs,
        )
