import torch
from torch import Tensor

from . import so2

import math


def uniform_grid(size: tuple[int, int], device: str | None = None) -> Tensor:
    """
    Returns a uniform grid of O2 elements where the rotations and
    reflections are both uniform SO2 grids. O2 elements are represented
    using a 2D vector where the first element denotes the reflection
    coefficient (1 or -1) and the second element denotes the rotation angle
    in the range `[0, 2pi)`.

    Arguments:
        - size: int denoting the number of elements in the grid.
        - device: optional str denoting the device.

    Returns:
        Tensor of shape (sum(size), 2).
    """
    angles = torch.cat(
        (
            so2.uniform_grid(size[0], device=device),
            so2.uniform_grid(size[1], device=device),
        ),
        dim=0,
    )

    coeffs = torch.cat(
        (torch.ones(size[0], 1, device=device), -torch.ones(size[1], 1, device=device)),
        dim=0,
    )

    return torch.cat((coeffs, angles), dim=-1)


def random_grid(size: int, device: str | None = None) -> Tensor:
    """
    Returns a uniform grid of O2 elements where the rotations and
    reflections are both uniform SO2 grids. O2 elements are represented
    using a 2D vector where the first element denotes the reflection
    coefficient (1 or -1) and the second element denotes the rotation angle
    in the range `[0, 2pi)`.

    Arguments:
        - size: int denoting the number of elements in the grid.
        - device: optional str denoting the device.

    Returns:
        Tensor of shape (sum(size), 2).
    """
    angles = so2.random_grid(size, device=device)

    coeffs = (2 * (torch.rand(size, 1) > 0.5)) - 1

    return torch.cat((coeffs, angles), dim=-1)


def inverse(R: Tensor) -> Tensor:
    """
    Returns the inverse reflection elements.

    Arguments:
        - R: tensor of shape (..., 2).

    Returns:
        Tensor of shape (..., 2) of inverse O2.
    """
    R = R.clone()
    R[..., 1] *= -1
    return R


def det(R: Tensor) -> Tensor:
    """
    Returns the determinants of R.

    Arguments:
        R: tensor of shape (..., 2).

    Returns:
        Tensor of shape (..., 1)
    """
    dims = R.shape[:-1]
    return R[..., 0].view(*dims, 1)


def left_apply_angle(R1: Tensor, R2: Tensor) -> Tensor:
    """
    Returns the left group action of R1 applied to R2.

    Arguments:
        - R1: tensor of shape (..., 2).
        - R2: tensor of shape (..., 2).

    Returns:
        Tensor containing left group action.
    """
    coeffs = R1[..., 0] * R2[..., 0]
    angles = (R1[..., 1] + R2[..., 1]) % (2 * math.pi)

    return torch.stack((coeffs, angles), dim=-1)


def left_apply_to_angle(R1: Tensor, R2: Tensor) -> Tensor:
    """
    Applies the left group action of each element in R1 to
    each element in R2.

    Arguments:
        - R1: tensor of shape (N 2).
        - R2: tensor of shape (M, 2).

    Returns:
        Tensor of shape (N, M, 2)
    """
    coeffs = R1[:, None, 0] * R2[:, 0]
    angles = (R1[:, None, 1] + R2[:, 1]) % (2 * math.pi)

    return torch.stack((coeffs, angles), dim=-1)


def left_apply_to_R2(R: Tensor, grid: Tensor) -> Tensor:
    """
    Applies every O3 in R, parameterized as angles,
    to every vector in grid.

    Arguments:
        - R: tensor of shape `(N, 2)` of O3 elements.
        - grid: tensor of shape (W, H, 2) of vectors.

    Returns:
        Tensor of shape `(N, W, H, 2)`.
    """
    coeffs, angles = R[:, 0], R[:, 1]

    return coeffs.view(-1, 1, 1, 1) * so2.left_apply_angle_to_R2(
        angles.view(-1, 1), grid
    )


def grid_sample(
    grid: Tensor,
    signal: Tensor,
    signal_grid: Tensor,
    signal_grid_size: tuple[int, int],
    mode: str = "rbf",
    rotation_width: float = 0.5,
    reflection_width: float = 0.5,
) -> Tensor:
    """
    Samples given O3 grid based on gived reference signal and
    corresponding signal grid.

    NOTE: It is assumed the signal and grid are ordered based on
    rotations first, then reflections. Order of rotations and reflections
    in input grid does not matter.

    Arguments:
        grid: Tensor of shape `(N, 10)` of O3 elements.
        signal: Tensor of shape `(M, S)`.
        signal_grid: Tensor of shape `(M, 3, 3)` of corresponding
                     rotation elements.
        signal_grid_size: Tuple of `(n_rotations, n_reflections)`
                          where n_rotations + n_reflections = M.
        mode: Interpolation mode used, supports "rbf" (default) and
              "nearest".
        width: Width for RBF kernel when using "rbf mode.
    """
    n_rotations, n_reflections = signal_grid_size

    coeffs, R = grid[:, 0], grid[:, 1].view(-1, 1)

    so2_signal = signal[:n_rotations]
    so2_signal_grid = signal_grid[:n_rotations, 1].unsqueeze(-1)

    r_signal = signal[n_rotations:]
    r_signal_grid = signal_grid[n_rotations:, 1].unsqueeze(-1)

    # find rotations and reflections
    so2_idx = torch.where(coeffs == 1)[0]
    r_idx = torch.where(coeffs == -1)[0]

    # sample rotations and reflections if they exist
    if n_rotations:
        so2_signal = so2.grid_sample(
            R[so2_idx], so2_signal, so2_signal_grid, mode=mode, width=rotation_width
        )

    if n_reflections:
        r_signal = so2.grid_sample(
            R[r_idx], r_signal, r_signal_grid, mode=mode, width=reflection_width
        )

    sampled_signal = torch.cat((so2_signal, r_signal), dim=0)

    # restore original order of input  O3 grid
    perms = torch.argsort((torch.cat((so2_idx, r_idx), dim=0)))

    return sampled_signal[perms]
