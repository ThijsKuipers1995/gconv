import torch
from torch import Tensor


from torch.nn.functional import grid_sample


def create_grid_R3(size: int) -> Tensor:
    """
    Creates normalized coordinate grid of size
    (size, size, size, 3) containing normalized x, y, z position
    for each voxel (Depth, Width, Height).

    Arguments:
        - size: Integer denoting size of grid.

    Returns:
        - Tensor containing R3 grid of shape (size, size, size, 3).
    """
    x = torch.linspace(-1, 1, size)
    X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")

    return torch.stack((Z, Y, X), dim=-1)
