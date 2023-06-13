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

from torch.nn import functional as F


class GLiftingKernelSE3(GLiftingKernel):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        group_size,
        groups: int = 1,
        mode: str = "bilinear",
        padding_mode: str = "border",
        mask: bool = True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            gF.create_grid_R3(kernel_size),
            groups,
        )

        self.group_size = group_size
        self.mode = mode
        self.padding_mode = padding_mode

        mask = gF.create_spherical_mask(kernel_size) if mask else None
        self.register_buffer("mask", mask)

    def forward(self, H):
        H_product = R.left_apply_to_R3(R.matrix_inverse(H), self.grid_Rn)

        weight = F.grid_sample(
            self.weight.repeat_interleave(H.shape[0], dim=0),
            H_product.repeat(self.out_channels, 1, 1, 1, 1),
            mode=self.mode,
            padding_mode=self.padding_mode,
        ).view(
            self.out_channels,
            H.shape[0],
            self.in_channels // self.groups,
            *self.kernel_size,
        )

        return weight if self.mask is None else self.mask * weight


class GSeparableKernelSE3(GSeparableKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_size: int,
        groups: int = 1,
        group_mode: str = "rbf",
        group_mode_width: float = 0.0,
        spatial_mode: str = "bilinear",
        spatial_padding_mode: str = "border",
        mask: bool = True,
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
            (kernel_size, kernel_size, kernel_size),
            grid_H,
            gF.create_grid_R3(kernel_size),
            groups,
        )

        self.group_size = group_size
        self.group_mode = group_mode
        self.group_mode_width = group_mode_width
        self.spatial_mode = spatial_mode
        self.spatial_padding_mode = spatial_padding_mode

        mask = gF.create_spherical_mask(kernel_size) if mask else None
        self.register_buffer("mask", mask)

    def forward(self, in_H: Tensor, out_H: Tensor) -> Tensor:
        num_in_H, num_out_H = in_H.shape[0], out_H.shape[0]

        out_H_inverse = R.matrix_inverse(out_H)

        H_product_H = R.left_apply_to_matrix(out_H_inverse, in_H)
        H_product_Rn = R.left_apply_to_R3(out_H_inverse, self.grid_Rn)

        # interpolate SO3
        weight_H = (
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

        # interpolate R3
        weight_Rn = F.grid_sample(
            self.weight.repeat_interleave(num_out_H, dim=0),
            H_product_Rn.repeat(self.out_channels, 1, 1, 1, 1),
            mode=self.mode,
            padding_mode=self.padding_mode,
        )

        return weight_H, weight_Rn


class GKernelSE3(GKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_size: int,
        groups: int = 1,
        group_mode: str = "rbf",
        group_mode_width: float = 0.0,
        spatial_mode: str = "bilinear",
        spatial_padding_mode: str = "border",
        mask: bool = True,
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
            (kernel_size, kernel_size, kernel_size),
            grid_H,
            gF.create_grid_R3(kernel_size),
            groups,
        )

        self.group_size = group_size
        self.group_mode = group_mode
        self.group_mode_width = group_mode_width
        self.spatial_mode = spatial_mode
        self.spatial_padding_mode = spatial_padding_mode

        mask = gF.create_spherical_mask(kernel_size) if mask else None
        self.register_buffer("mask", mask)

    def forward(self, in_H: Tensor, out_H: Tensor) -> Tensor:
        num_in_H, num_out_H = in_H.shape[0], out_H.shape[0]

        out_H_inverse = R.matrix_inverse(out_H)

        H_product_H = R.left_apply_to_matrix(out_H_inverse, in_H)
        H_product_Rn = R.left_apply_to_R3(out_H_inverse, self.grid_Rn)

        # interpolate SO3
        weight = gF.so3_sample(
            H_product_H.view(-1, 3, 3),
            self.grid_H,
            self.weight.transpose(0, 2).reshape(self.num_H, -1),
            mode=self.group_mode,
            width=self.group_mode_width,
        ).view(
            num_out_H * num_in_H,
            (self.in_channels // self.groups) * self.out_channels,
            *self.kernel_size,
        )

        # interpolate R3
        weight = (
            F.grid_sample(
                weight,
                H_product_Rn.repeat_interleave(num_in_H, dim=0),
                mode=self.spatial_mode,
                padding_mode=self.spatial_padding_mode,
            )
            .view(
                num_out_H,
                num_in_H,
                self.in_channels // self.groups,
                self.out_channels,
                *self.kernel_size,
            )
            .transpose(0, 3)
            .transpose(1, 3)
        )

        return weight
