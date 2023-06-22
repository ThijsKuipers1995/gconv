from typing import Optional
import torch
from torch import Tensor

from . import so3


def uniform_grid(
    size: tuple[int, int], matrix_only: bool = False, device: Optional[str] = None
) -> Tensor:
    """
    Creates a grid of uniform rotations and reflections. Each O3 element
    is represented as a 10 dimensional vector, where the first element
    denotes the reflection coefficient, i.e., 1 or -1, and the remaining 9
    elements denote the flattened rotation matrix.

    Alternatively, if matrix_only is true, the grids will only consists of
    the rotation matrices, i.e., elements have a shape of 3 by 3.

    Arguments:
        - size: Tuple denoting `(n_rotations, n_reflections)`.
        - matrix_only: If true, will only rotation matrix part of O3 grid.

    Returns:
        Tensor of shape `(n_rotations + n_reflections, 10)` or
        `(n_rotations + n_reflections, 3, 3)` if matrix_only is true.
    """
    n_rotations, n_reflections = size

    R1 = so3.uniform_grid(n_rotations, "matrix", device=device)
    R2 = so3.uniform_grid(n_reflections, "matrix", device=device)
    R = torch.cat((R1, R2), dim=-0)

    if matrix_only:
        return R

    coeff1 = torch.ones(n_rotations, 1, device=device)
    coeff2 = -1 * torch.ones(n_reflections, 1, device=device)
    coeffs = torch.cat((coeff1, coeff2), dim=0)

    grid = torch.cat((coeffs, R.flatten(-2, -1)), dim=-1)

    return grid


def random(size: int | tuple, device: Optional[str] = None) -> Tensor:
    """
    Returns uniform randomly sampled O3 elements.

    Arguments:
        - size: int or tuple denoting the size.
        - device: device on which the tensor should be generated.

    Returns:
        Tensor of shape size of random O3 elements.
    """
    R = so3.random_matrix(size, device=device).flatten(-2, -1)
    coeffs = ((torch.rand(R.shape[:-2]) > 0.5) * 2) - 1

    return torch.cat((coeffs, R), dim=-1)


def left_apply_O3(H1: Tensor, H2: Tensor) -> Tensor:
    """
    Implements group product between H1 and H2, following
    default broadcasting rules.

    Arguments:
        - H1: Tensor of shape `(..., 10)`.
        - H2: Tensor of shape `(..., 10)`.

    Returns:
        Tensor of shape (..., 10).
    """
    R1 = H1[..., 1:].unflatten(-1, (3, 3))
    R2 = H2[..., 1:].unflatten(-1, (3, 3))

    coeff1 = H1[..., 0].unsqueeze(-1)
    coeff2 = H2[..., 0].unsqueeze(-1)

    R = (R1 @ R2).flatten(-2, -1)
    coeff = coeff1 * coeff2

    return torch.cat((coeff, R), dim=-1)


def left_apply_to_O3(H1: Tensor, H2: Tensor) -> Tensor:
    """
    Implements pairwise O3 group product, applying every
    element in H1 to every element in H2.

    Arguments:
        - H1: Tensor of shape `(N, 10)`.
        - H2: Tensor of shape `(M, 10)`.

    Returns:
        Tensor of shape (N, M, 10).
    """
    R1 = H1[:, 1:].view(-1, 3, 3)
    R2 = H2[:, 1:].view(-1, 3, 3)

    coeff1 = H1[:, 0].unsqueeze(-1)
    coeff2 = H2[:, 0].unsqueeze(-1)

    R = (R1[:, None] @ R2).flatten(-2, -1)
    coeff = coeff1[:, None] * coeff2

    return torch.cat((coeff, R), dim=-1)


def left_apply_to_R3(H: Tensor, grid: Tensor) -> Tensor:
    """
    Applies each O3 element in `H` to `grid`.

    Arguments:
        - H: Tensor of shape `(..., 10)` of O3 elements.
        - grid: Tensor of shape `(x, y, z, 3)` of R3 vectors.

    Returns:
        - Tensor of shape `(..., x, y, z, 3)` of transformed
          R3 vectors.
    """
    coeffs, R = H[..., 0][..., None, None, None, None], H[..., 1:].unflatten(-1, (3, 3))
    return coeffs * (R[..., None, None, None, :, :] @ grid[..., None]).squeeze(-1)


def det(H: Tensor) -> Tensor:
    """
    Returns the determinants of the given O3 elements.

    Arguments:
        - H: Tensor of shape (..., 10).

    Returns:
        Tensor of shape (..., 1) of determinants.
    """
    return H[..., 0].unsqueeze(-1)


def inverse(H: Tensor) -> Tensor:
    """
    Returns the inverse of the given group elements.

    Arguments
        - H: Tensor of shape (..., 10).

    Returns:
        Tensor of shape (..., 10) of inverted elements.
    """
    coeffs, R = H[..., 0, None], H[..., 1:].unflatten(-1, (3, 3))
    return torch.cat((coeffs, R.mT.flatten(-2, -1)), dim=-1)


def grid_sample(
    grid: Tensor,
    signal: Tensor,
    signal_grid: Tensor,
    signal_grid_size: tuple[int, int],
    mode: str = "rbf",
    rotation_width: float = 0.5,
    reflection_width: float = 0.5,
):
    """
    Samples given O3 grid based on gived reference signal and
    corresponding signal grid.

    NOTE: It is assumed the signal and grid are ordered based on
    rotations first, then reflections. Order of rotations and reflections
    in input grid does not matter.

    Arguments:
        grid: Tensor of shape `(N, 10)` of O3 elements.
        signal: Tensor of shape `(M, S)`.
        signal_grid: Tensor of shape `(M, 10)` of corresponding
                     rotation elements.
        signal_grid_size: Tuple of `(n_rotations, n_reflections)`
                          where n_rotations + n_reflections = M.
        mode: Interpolation mode used, supports "rbf" (default) and
              "nearest".
        width: Width for RBF kernel when using "rbf mode.
    """
    n_rotations, n_reflections = signal_grid_size

    coeffs, R = grid[:, 0], grid[:, 1:].view(-1, 3, 3)

    so3_signal = signal[:n_rotations]
    so3_signal_grid = signal_grid[:n_rotations, 1:].view(-1, 3, 3)

    r_signal = signal[n_rotations:]
    r_signal_grid = signal_grid[n_rotations:, 1:].view(-1, 3, 3)

    # find rotations and reflections
    so3_idx = torch.where(coeffs == 1)[0]
    r_idx = torch.where(coeffs == -1)[0]

    # sample rotations and reflections if they exist
    if n_rotations:
        so3_signal = so3.grid_sample(
            R[so3_idx], so3_signal, so3_signal_grid, mode=mode, width=rotation_width
        )

    if n_reflections:
        r_signal = so3.grid_sample(
            R[r_idx], r_signal, r_signal_grid, mode=mode, width=reflection_width
        )

    sampled_signal = torch.cat((so3_signal, r_signal), dim=0)

    # restore original order of input  O3 grid
    perms = torch.argsort(torch.cat((so3_idx, r_idx)), dim=0)

    return sampled_signal[perms]
