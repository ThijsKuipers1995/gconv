import torch.nn as nn

from torch import Tensor


class GBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        N, C, _H, Z, *dims = x.shape
        return super().forward(x.view(N, C, _H * Z, *dims)).view(N, C, _H, Z, *dims), H


class GBatchNorm3d(nn.BatchNorm3d):
    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        N, C, _H, Z, *dims = x.shape
        return super().forward(x.view(N, C, _H * Z, *dims)).view(N, C, _H, Z, *dims), H


class GLayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        return super().forward(x), H


class GInstanceNorm2d(nn.InstanceNorm2d):
    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        N, C, _H, Z, *dims = x.shape
        return super().forward(x.view(N, C, _H * Z, *dims)).view(N, C, _H, Z, *dims), H


class GInstanceNorm3d(nn.InstanceNorm3d):
    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        N, C, _H, Z, *dims = x.shape
        return super().forward(x.view(N, C, _H * Z, *dims)).view(N, C, _H, Z, *dims), H
