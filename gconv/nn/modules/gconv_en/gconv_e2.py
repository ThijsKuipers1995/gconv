from typing import Optional
from gconv.nn.kernels import GLiftingKernelE2, GSeparableKernelE2, GKernelE2
from gconv.nn.modules.gconv import GLiftingConv2d, GSeparableConv2d, GConv2d

from gconv.geometry import o2

from torch import Tensor


class GLiftingConvE2(GLiftingConv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        group_kernel_size: tuple | int = (4, 4),
        grid_H: Optional[Tensor] = None,
        padding_mode: str = "zeros",
        permute_output_grid: bool = True,
        sampling_mode="bilinear",
        sampling_padding_mode="border",
        bias: bool = False,
        mask: bool = True,
    ) -> None:
        """
        Implements E3 separable group convolution.

        Arguments:
            - int_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: tuple denoting the spatial kernel size.
            - groups: int denoting the number of groups for depth-wise separability.
            - stride: int denoting the stride.
            - padding: int or denoting padding.
            - dilation: int denoting dilation.
            - group_kernel_size: tuple or int denoting the group kernel size (default (4, 4)).
                                 If tuple, (x, y) denotes the rotation and reflection subgroup
                                 sizes, respectively. If int, size will be used for both rotation
                                 and reflection subgroup kernels.
            - grid_H: tensor of shape (N, 3, 3) of SO3 elements (rotation matrices). If
                      not provided, a uniform grid will be initalizd of size group_kernel_size.
                      If provided, group_kernel_size will be set to N.
            - padding_mode: str denoting the padding mode.
            - permute_output_grid: bool that if true will randomly permute output group grid
                                   for estimating continuous groups.
            - spatial_sampling_mode: str denoting mode used for sampling spatial weights. Supports
                                     bilinear (default) or nearest.
            - spatial_sampling_padding_mode: str denoting padding mode for spatial weight sampling,
                                             border (default) is recommended.
            - bias: bool that if true, will initialzie bias parameters.
            - mask: bool that if true, will initialize spherical mask applied to spatial weights.
        """
        kernel = GLiftingKernelE2(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size=group_kernel_size,
            groups=groups,
            sampling_mode=sampling_mode,
            sampling_padding_mode=sampling_padding_mode,
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
            H = o2.left_apply_angle(o2.random_grid(1, device=input.device), H)

        return super().forward(input, H)


class GSeparableConvE3(GSeparableConv2d):
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
        group_rotation_sampling_width: float = 0.0,
        group_reflection_sampling_width: float = 0.0,
        spatial_sampling_mode: str = "bilinear",
        spatial_sampling_padding_mode: str = "border",
        mask: bool = True,
        bias: bool = False,
        grid_H: Optional[Tensor] = None,
    ) -> None:
        """
        Implements E3 separable group convolution.

        Arguments:
            - int_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: tuple denoting the spatial kernel size.
            - groups: int denoting the number of groups for depth-wise separability.
            - stride: int denoting the stride.
            - padding: int or denoting padding.
            - dilation: int denoting dilation.
            - group_kernel_size: int denoting the group kernel size (default 4).
            - grid_H: tensor of shape (N, 3, 3) of SO3 elements (rotation matrices). If
                      not provided, a uniform grid will be initalizd of size group_kernel_size.
                      If provided, group_kernel_size will be set to N.
            - padding_mode: str denoting the padding mode.
            - permute_output_grid: bool that if true will randomly permute output group grid
                                   for estimating continuous groups.
            - group_sampling_mode: str denoting mode used for sampling group weights. Supports
                                   rbf (default) or nearest.
            - group_rotation_sampling_width: float denoting width of Gaussian rbf kernel when using rbf sampling
                                             for rotation subgroup kernel. If set to 0.0 (default, recommended),
                                             width will be initialized on the density of grid_H.
            - group_reflection_sampling_width: float denoting width of Gaussian rbf kernel when using rbf sampling
                                               for reflection subgroup kernel. If set to 0.0 (default, recommended),
                                               width will be initialized on the density of grid_H.
            - spatial_sampling_mode: str denoting mode used for sampling spatial weights. Supports
                                     bilinear (default) or nearest.
            - spatial_sampling_padding_mode: str denoting padding mode for spatial weight sampling,
                                             border (default) is recommended.
            - bias: bool that if true, will initialzie bias parameters.
            - mask: bool that if true, will initialize spherical mask applied to spatial weights.
        """
        kernel = GSeparableKernelE2(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size,
            groups=groups,
            group_mode=group_sampling_mode,
            group_rotation_sampling_width=group_rotation_sampling_width,
            group_reflection_sampling_width=group_reflection_sampling_width,
            spatial_sampling_mode=spatial_sampling_mode,
            spatial_sampling_padding_mode=spatial_sampling_padding_mode,
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
            H = o2.left_apply_angle(o2.random_grid(1, device=input.device), H)

        return super().forward(input, H)


class GConvE3(GConv2d):
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
        group_rotation_sampling_width: float = 0.0,
        group_reflection_sampling_width: float = 0.0,
        spatial_sampling_mode: str = "bilinear",
        spatial_sampling_padding_mode: str = "border",
        mask: bool = True,
        bias: bool = False,
        grid_H: Optional[Tensor] = None,
    ) -> None:
        """
        Implements E3 separable group convolution.

        Arguments:
            - int_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: tuple denoting the spatial kernel size.
            - groups: int denoting the number of groups for depth-wise separability.
            - stride: int denoting the stride.
            - padding: int or denoting padding.
            - dilation: int denoting dilation.
            - group_kernel_size: int denoting the group kernel size (default 4).
            - grid_H: tensor of shape (N, 3, 3) of SO3 elements (rotation matrices). If
                      not provided, a uniform grid will be initalizd of size group_kernel_size.
                      If provided, group_kernel_size will be set to N.
            - padding_mode: str denoting the padding mode.
            - permute_output_grid: bool that if true will randomly permute output group grid
                                   for estimating continuous groups.
            - group_sampling_mode: str denoting mode used for sampling group weights. Supports
                                   rbf (default) or nearest.
            - group_rotation_sampling_width: float denoting width of Gaussian rbf kernel when using rbf sampling
                                             for rotation subgroup kernel. If set to 0.0 (default, recommended),
                                             width will be initialized on the density of grid_H.
            - group_reflection_sampling_width: float denoting width of Gaussian rbf kernel when using rbf sampling
                                               for reflection subgroup kernel. If set to 0.0 (default, recommended),
                                               width will be initialized on the density of grid_H.
            - spatial_sampling_mode: str denoting mode used for sampling spatial weights. Supports
                                     bilinear (default) or nearest.
            - spatial_sampling_padding_mode: str denoting padding mode for spatial weight sampling,
                                             border (default) is recommended.
            - bias: bool that if true, will initialzie bias parameters.
            - mask: bool that if true, will initialize spherical mask applied to spatial weights.
        """
        kernel = GKernelE2(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size,
            groups=groups,
            group_mode=group_sampling_mode,
            group_rotation_sampling_width=group_rotation_sampling_width,
            group_reflection_sampling_width=group_reflection_sampling_width,
            spatial_sampling_mode=spatial_sampling_mode,
            spatial_sampling_padding_mode=spatial_sampling_padding_mode,
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
            H = o2.left_apply_angle(o2.random_grid(1, device=input.device), H)

        return super().forward(input, H)
