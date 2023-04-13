import torch
from torch import Tensor

from gconv.geometry import so3
from gconv.geometry import interpolation

import gconv.nn.functional._grid_cache as grid_cache

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
    Z, Y, X = torch.meshgrid(x, x, x, indexing="ij")

    return torch.stack((X, Y, Z), dim=-1)


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
        grid = so3.uniform_grid(
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
    steps: int = 500,
    cache_grid: bool = True,
) -> Tensor:
    if type.lower() == "eye":
        grid = so3.identity(device)
    if type.lower() == "k" or type.lower() == "klein":
        grid = so3.klein_group(device)
    if type.lower() == "t" or type.lower() == "tetrahedral":
        grid = so3.tetrahedral(device)
    if type.lower() == "o" or type.lower() == "octahedral":
        grid = so3.octahedral(device)
    if type.lower() == "i" or type.lower() == "icosahedral":
        grid = so3.icosahedral(device)
    if type.lower() == "u":
        return _create_uniform_grid(
            size,
            "so3",
            parameterization,
            steps=steps,
            device=device,
            cache_grid=cache_grid,
        )
    if type.lower() == "r":
        grid = so3.random_quat(size, device)

    return grid if parameterization.lower() == "quat" else so3.quat_to_matrix(grid)


def grid_sample(
    signal: Tensor, grid: Tensor, mode: str = "bilinear", padding_mode="border"
) -> Tensor:
    """
    Samples R2 or R3 signal on provided signal for given grid.

    Arguments:
        - signal: Tensor of shape `(N, C, W, H)` or `(N, C, D, W, H)` containing 2D or 3D signal.
        - grid: Tensor of shape `(G, W, H, 2)` or `(G, D, W, H, 3)` of `D` grids to interpolate.
                This grid is a flowfield, defined from -1 to 1 (see PyTorch nn.functional.grid_sample documentation).
        - mode: Interpolation mode.
        - padding_mode: Padding mode used for values outside grid boundary
    """
    return F.grid_sample(
        signal.repeat_interleave(grid.shape[0], dim=0),
        grid.repeat(signal.shape[0], *((grid.shape[-1] + 1) * (1,))),
        mode=mode,
        padding_mode=padding_mode,
    ).view(signal.shape[0], grid.shape[0], signal.shape[1], *signal.shape[2:])


def so3_sample(
    signal: Tensor,
    reference_grid: Tensor,
    grid: Tensor,
    mode: str = "rbf",
    width: float = 0.1,
) -> Tensor:
    """
    Samples SO3 signal on provided signal and corresponding SO3 reference grid
    for the given grid of SO3 elements.

    Arguments:
        - signal: SO3 signal to interpolate of shape `(N, H1, 4)`.
        - reference_grid: Reference grid corresponding to signal of shape `(N, H, 4)`.
        - grid: Grid of SO3 elements to sample of shape `(N, H2, 4)`.
        - width: Width used for RBF interpolation kernel.

    Returns:
        - Tensor of shape (N, H2, S) containing sampled signal.
    """
    if mode == "nearest":
        return interpolation.interpolate_NN(
            grid, reference_grid, signal, dist_fn=so3.geodesic_distance
        )
    if mode == "bcc":
        return interpolation.interpolate_BCC(
            grid, reference_grid, signal, dist_fn=so3.geodesic_distance
        )
    if mode == "rbf":
        return interpolation.interpolate_RBF(
            grid, reference_grid, signal, dist_fn=so3.geodesic_distance, width=width
        )
    raise ValueError(f"unknown interpolation mode `{mode=}`.")
