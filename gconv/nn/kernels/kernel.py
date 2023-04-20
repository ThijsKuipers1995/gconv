from typing import Any
from pyparsing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init

from torch import Tensor

import math

from gconv.geometry import so3
from gconv.nn import functional as gF


class _KernelBase(nn.Module):
    pass


class GKernel(_KernelBase):
    pass


class GSeparableKernel(_KernelBase):
    pass


class LiftingKernel(_KernelBase):
    pass
