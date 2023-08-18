import torch

from .grid import create_grid_R3, create_grid_R2
from torch import Tensor


def create_spherical_mask_R3(size: int, device: str = None) -> Tensor:
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


def create_spherical_mask_R2(size: int, device: str = None) -> Tensor:
    """
    Creates spherical mask of size (size, size).

    Arguments:
        - size: Size of mask.
        - device: Device on which mask is created.

    Returns:
        - Tensor of shape (size, size).
    """
    grid = create_grid_R2(size).to(device)

    return (torch.linalg.norm(grid, dim=-1) < 1.1).float()
