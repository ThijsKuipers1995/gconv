import torch

from .grid import create_grid_R3
from torch import Tensor


def _create_spherical_mask(size: int, device: str = None) -> Tensor:
    """
    Creates spherical mask of size (size, size, size).

    Arguments:
        - size: Size of mask.
        - device: Device on which mask is created.

    Returns:
        - Tensor of shape (size, size, size).
    """
    grid = create_grid_R3(size).to(device)

    return (torch.linalg.norm(grid, dim=-1) < 1.1).float()
