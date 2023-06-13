"""
kernel_so3.py

Implements a kernel for SO3 convolutions.
"""
from __future__ import annotations
from typing import Optional

from gconv3d.nn.kernels import GKernel

from torch import Tensor


from gconv3d.geometry import rotation as R
from gconv3d.nn import functional as gF


class GKernelSO3(GKernel):
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

        super().__init__(
            in_channels,
            out_channels,
            (1, 1, 1),
            grid_H,
            None,
            groups,
        )

        self.group_size = group_size
        self.mode = mode
        self.width = width

    def forward(self, in_H: Tensor, out_H: Tensor) -> Tensor:
        num_in_H, num_out_H = in_H.shape[0], out_H.shape[0]

        out_H_inverse = R.matrix_inverse(out_H)

        H_product_H = R.left_apply_to_matrix(out_H_inverse, in_H)

        # interpolate SO3
        weight = (
            gF.so3_sample(
                H_product_H.view(-1, 3, 3),
                self.weight.transpose(0, 2).reshape(1, self.num_H, -1),
                mode=self.mode,
                width=self.width,
            )
            .view(
                num_in_H,
                num_out_H,
                self.in_channels // self.groups,
                self.out_channels,
                *self.kernel_size,
            )
            .transpose(0, 3)
            .transpose(1, 3)
        )

        return weight
