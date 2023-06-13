import torch
from torch import Tensor

from gconv3d.geometry import rotation as R
from gconv3d.geometry import interpolation

import gconv3d.nn.functional._grid_cache as grid_cache

import torch.nn.functional as F


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
    grid = torch.meshgrid(x, x, x, indexing="ij")

    return torch.stack(reversed(grid), dim=-1)


def _create_uniform_grid(
    size: int,
    grid_type: str,
    parameterization: str,
    steps: int,
    device: str = None,
    cache_grid: bool = True,
) -> Tensor:
    try:
        return grid_cache.load_grid(size, grid_type, parameterization)
    except KeyError:
        grid = R.uniform_grid(
            size, steps=steps, device=device, parameterization=parameterization
        )

        if cache_grid:
            grid_cache.save_grid(grid, grid_type, parameterization)

    return grid


def create_grid_SO3(
    type: str,
    size: int = 1,
    parameterization: str = "quat",
    device: str = None,
    steps: int = 1000,
    cache_grid: bool = True,
) -> Tensor:
    type = type.lower()

    if type == "eye":
        grid = R.identity(device)
    if type == "k" or type == "klein":
        grid = R.klein_group(device)
    if type == "t" or type == "tetrahedral":
        grid = R.tetrahedral(device)
    if type == "o" or type == "octahedral":
        grid = R.octahedral(device)
    if type == "i" or type == "icosahedral":
        grid = R.icosahedral(device)
    if type == "u":
        return _create_uniform_grid(
            size,
            "so3",
            parameterization.lower(),
            steps=steps,
            device=device,
            cache_grid=cache_grid,
        )
    if type == "r":
        grid = R.random_quat(size, device)

    return grid if parameterization.lower() == "quat" else R.quat_to_matrix(grid)


def grid_sample(
    signal: Tensor, grid: Tensor, mode: str = "bilinear", padding_mode="border"
) -> Tensor:
    """
    TODO: arguments (see F.grid_sample)

    Returns:
        Tensor of shape `(N, G, C, W, H)` or `(N, G, C, D, W, H)`.
    """
    return F.grid_sample(
        signal,
        grid,
        mode=mode,
        padding_mode=padding_mode,
    ).view(signal.shape[0], grid.shape[0], signal.shape[1], *signal.shape[2:])


def so3_sample(
    grid: Tensor,
    signal: Tensor,
    signal_grid: Tensor,
    mode: str = "rbf",
    width: float = 0.5,
) -> Tensor:
    """
    Samples SO3 signal on provided signal and corresponding SO3 signal grid
    for the given grid of SO3 elements. Supports both matrix and euler
    parameterizations.

    Arguments:
        - signal: SO3 signal to interpolate of shape `(H1, S)`.
        - signal_grid: signal grid corresponding to signal of shape `(H1, 4)`.
        - grid: Grid of SO3 elements to sample of shape `(H2, 4)`.
        - width: Width used for RBF interpolation kernel.

    Returns:
        - Tensor of shape (H2, S) containing sampled signal.
    """
    # If input are matrices, convert to quats
    if grid.ndim == 3:
        grid = R.matrix_to_quat(grid)
        signal_grid = R.matrix_to_quat(signal_grid)

    if mode == "nearest":
        return interpolation.interpolate_NN(
            grid[None],
            signal_grid[None],
            signal[None],
            dist_fn=R.geodesic_distance,
        ).squeeze(0)
    if mode == "rbf":
        return interpolation.interpolate_RBF(
            grid[None],
            signal_grid[None],
            signal[None],
            dist_fn=R.geodesic_distance,
            width=width,
        ).squeeze(0)
    raise ValueError(f"unknown interpolation mode `{mode=}`.")
