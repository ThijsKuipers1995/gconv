import torch.nn as nn
from torch import Tensor


class GWrap(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        """
        For wrapping any module that applies element-wise operations
        to the input to accept group convolution input.

        Applies `module(x)` during the forward pass.

        Arguments:
            - module: nn.Module.
        """
        self.module = module

    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        return self.module(x), H
        