import torch.nn as nn

from torch import Tensor


class GMaxSpatialPool2d(nn.MaxPool2d):
    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        return super().forward(x.flatten(1, 2)).view(*x.shape[:3]), H


class GMaxSpatialPool3d(nn.MaxPool3d):
    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        return super().forward(x.flatten(1, 2)).view(*x.shape[:3]), H


class GMaxGroupPool(nn.Module):
    def forward(self, x: Tensor, _: Tensor) -> Tensor:
        return x.max(dim=2)


class GMaxGlobalPool(nn.Module):
    def forward(self, x: Tensor, _: Tensor) -> Tensor:
        x.flatten(2, -1).max(-1, keepdim=True)


class GAvgSpatialPool2d(nn.AvgPool2d):
    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        return super().forward(x.flatten(1, 2)).view(*x.shape[:3]), H


class GAvgSpatialPool3d(nn.AvgPool3d):
    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        return super().forward(x.flatten(1, 2)).view(*x.shape[:3]), H


class GAvgGroupPool(nn.Module):
    def forward(self, x: Tensor, _: Tensor) -> Tensor:
        return x.mean(dim=2)


class GAvgGlobalPool(nn.Module):
    def forward(self, x: Tensor, _: Tensor) -> Tensor:
        x.flatten(2, -1).mean(-1, keepdim=True)


class GAdaptiveMaxSpatialPool2d(nn.AdaptiveMaxPool2d):
    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        return super().forward(x.flatten(1, 2)).view(*x.shape[:3]), H


class GAdaptiveMaxSpatialPool3d(nn.AdaptiveMaxPool3d):
    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        return super().forward(x.flatten(1, 2)).view(*x.shape[:3]), H


class GAdaptiveAvgSpatialPool2d(nn.AdaptiveAvgPool2d):
    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        return super().forward(x.flatten(1, 2)).view(*x.shape[:3]), H


class GAdaptiveAvgSpatialPool3d(nn.AdaptiveAvgPool3d):
    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        return super().forward(x.flatten(1, 2)).view(*x.shape[:3]), H
