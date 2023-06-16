from typing import Optional
from torch import Tensor

from .gconv import GLiftingConv3d, GSeparableConv3d, GConv3d
from gconv3d.nn.kernels import GLiftingKernelSE3, GSeparableKernelSE3, GKernelSE3

from gconv3d.geometry import so3


class GLiftingConvSE3(GLiftingConv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        group_kernel_size: int = 4,
        grid_H: Optional[Tensor] = None,
        padding_mode: str = "zeros",
        permute_output_grid: bool = True,
        sampling_mode="bilinear",
        sampling_padding_mode="border",
        bias: bool = False,
        mask: bool = True,
    ) -> None:
        kernel = GLiftingKernelSE3(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size=group_kernel_size,
            groups=groups,
            mode=sampling_mode,
            padding_mode=sampling_padding_mode,
            mask=mask,
            grid_H=grid_H,
        )

        self.permute_output_grid = permute_output_grid

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size,
            kernel,
            groups,
            stride,
            padding,
            dilation,
            padding_mode,
            bias,
        )

    def forward(
        self, input: Tensor, H: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        if H is None:
            H = self.kernel.grid_H

        if self.permute_output_grid:
            H = so3.left_apply_matrix(so3.random_matrix(1), H)

        return super().forward(input, H)


class GSeparableConvSE3(GSeparableConv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        group_kernel_size: int,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        permute_output_grid: bool = True,
        group_sampling_mode: str = "rbf",
        group_sampling_width: float = 0.0,
        spatial_sampling_mode: str = "bilinear",
        spatial_sampling_padding_mode: str = "border",
        mask: bool = True,
        bias: bool = False,
        grid_H: Optional[Tensor] = None,
    ) -> None:
        kernel = GSeparableKernelSE3(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size,
            groups=groups,
            group_mode=group_sampling_mode,
            group_mode_width=group_sampling_width,
            spatial_mode=spatial_sampling_mode,
            spatial_padding_mode=spatial_sampling_padding_mode,
            mask=mask,
            grid_H=grid_H,
        )

        self.permute_output_grid = permute_output_grid

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size,
            kernel,
            groups,
            stride,
            padding,
            dilation,
            padding_mode,
            bias,
        )

    def forward(
        self, input: Tensor, in_H: Tensor, out_H: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        if out_H is None:
            out_H = in_H

        if self.permute_output_grid:
            out_H = so3.left_apply_matrix(so3.random_matrix(1), out_H)

        return super().forward(input, in_H, out_H)


class GConvSE3(GConv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        group_kernel_size: int,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        permute_output_grid: bool = True,
        group_sampling_mode: str = "rbf",
        group_sampling_width: float = 0.0,
        spatial_sampling_mode: str = "bilinear",
        spatial_sampling_padding_mode: str = "border",
        mask: bool = True,
        bias: bool = False,
        grid_H: Optional[Tensor] = None,
    ) -> None:
        kernel = GKernelSE3(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            group_kernel_size=group_kernel_size,
            groups=groups,
            group_mode=group_sampling_mode,
            group_mode_width=group_sampling_width,
            spatial_mode=spatial_sampling_mode,
            spatial_padding_mode=spatial_sampling_padding_mode,
            mask=mask,
            grid_H=grid_H,
        )

        self.permute_output_grid = permute_output_grid

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size,
            kernel,
            groups,
            stride,
            padding,
            dilation,
            padding_mode,
            bias,
        )

    def forward(
        self, input: Tensor, in_H: Tensor, out_H: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        if out_H is None:
            out_H = in_H

        if self.permute_output_grid:
            out_H = so3.left_apply_matrix(so3.random_matrix(1), out_H)

        return super().forward(input, in_H, out_H)
