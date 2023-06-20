from typing import Optional
import torch.nn as nn

from torch import Tensor


class GMaxSpatialPool2d(nn.MaxPool2d):
    """
    Performs spatial max pooling on 2d spatial inputs.
    """

    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        return super().forward(x.flatten(1, 2)).view(*x.shape[:3]), H


class GMaxSpatialPool3d(nn.MaxPool3d):
    """
    Performs spatial max pooling on 3d spatial inputs.
    """

    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        return super().forward(x.flatten(1, 2)).view(*x.shape[:3]), H


class GMaxGroupPool(nn.Module):
    """
    Performs max pooling over the group dimension.
    """

    def forward(self, x: Tensor, _: Optional[Tensor] = None) -> Tensor:
        return x.max(dim=2)


class GMaxGlobalPool(nn.Module):
    """
    Performs global max pooling on group + spatial dimensions.
    """

    def forward(self, x: Tensor, _: Optional[Tensor] = None) -> Tensor:
        x.flatten(2, -1).max(-1, keepdim=True)


class GAvgSpatialPool2d(nn.AvgPool2d):
    """
    Performs mean spatial pooling on 2d spatial inputs.
    """

    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        return super().forward(x.flatten(1, 2)).view(*x.shape[:3]), H


class GAvgSpatialPool3d(nn.AvgPool3d):
    """
    Performs mean spatial pooling on 3d spatial inputs.
    """

    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        return super().forward(x.flatten(1, 2)).view(*x.shape[:3]), H


class GAvgGroupPool(nn.Module):
    """
    Performs mean spatial pooling over the group dimension.
    """

    def forward(self, x: Tensor, _: Optional[Tensor] = None) -> Tensor:
        return x.mean(dim=2)


class GAvgGlobalPool(nn.Module):
    """
    Performs mean global pooling over group + spatial dimensions.
    """

    def forward(self, x: Tensor, _: Optional[Tensor] = None) -> Tensor:
        x.flatten(2, -1).mean(-1, keepdim=True)


class GAdaptiveMaxSpatialPool2d(nn.AdaptiveMaxPool2d):
    """
    Performs adaptive max spatial pooling on 2d spatial inputs.
    """

    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        return super().forward(x.flatten(1, 2)).view(*x.shape[:3]), H


class GAdaptiveMaxSpatialPool3d(nn.AdaptiveMaxPool3d):
    """
    Performs adaptive max spatial pooling on 3d spatial inputs.
    """

    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        return super().forward(x.flatten(1, 2)).view(*x.shape[:3]), H


class GAdaptiveAvgSpatialPool2d(nn.AdaptiveAvgPool2d):
    """
    Performs adaptive mean spatial pooling on 2d spatial inputs.
    """

    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        return super().forward(x.flatten(1, 2)).view(*x.shape[:3]), H


class GAdaptiveAvgSpatialPool3d(nn.AdaptiveAvgPool3d):
    """
    Performs adaptive mean spatial pooling on 3d spatial inputs.
    """

    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        return super().forward(x.flatten(1, 2)).view(*x.shape[:3]), H
