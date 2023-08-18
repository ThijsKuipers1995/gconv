import torch.nn as nn
from torch import Tensor


class GReLU(nn.ReLU):
    def forward(self, x: Tensor, H: Tensor):
        return super().forward(x), H


class GGELU(nn.GELU):
    def forward(self, x: Tensor, H: Tensor):
        return super().forward(x), H


class GLeakyReLU(nn.LeakyReLU):
    def forward(self, x: Tensor, H: Tensor):
        return super().forward(x), H
