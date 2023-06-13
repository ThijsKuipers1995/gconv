"""
kernel_so3.py

Implements a kernel for SO3 convolutions.
"""
from __future__ import annotations

from typing import Optional

from gconv3d.nn.kernels import GSubgroupKernel

from torch import Tensor

from gconv3d.geometry import rotation as R
from gconv3d.nn import functional as gF


class GSubgroupKernelSO3(GSubgroupKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group_size: int,
        groups: int = 1,
        mode: str = "rbf",
        width: float = 0.0,
        grid_H: Optional[Tensor] = None,
    ) -> None:
        grid_H = (
            grid_H
            if grid_H is not None
            else gF.create_grid_SO3("uniform", size=group_size)
        )

        interpolate_H_kwargs = {"mode": mode, "width": width}

        super().__init__(
            in_channels,
            out_channels,
            (1, 1, 1),
            grid_H,
            None,
            groups,
            inverse_H=R.matrix_inverse,
            left_apply_to_H=R.left_apply_to_matrix,
            interpolate_H=gF.so3_sample,
            interpolate_H_kwargs=interpolate_H_kwargs,
        )
