import torch
from torch import Tensor

from gconv.geometry import interpolation

import math


def angle_to_matrix(R: Tensor) -> Tensor:
    """
    Converts tensor of angles to rotation matrices.

    Arguments:
        - R: tensor of shape `(..., 1)` of angles.

    Returns:
        Tensor of shape `(..., 2, 2)`.
    """
    matrices = R.new_empty((*R.shape[:-1], 2, 2))

    cos_H = torch.cos(R).flatten(-2)
    sin_H = torch.sin(R).flatten(-2)

    matrices[..., 0, 0] = cos_H
    matrices[..., 0, 1] = -sin_H
    matrices[..., 1, 0] = sin_H
    matrices[..., 1, 1] = cos_H

    return matrices


def angle_to_euclid(R: Tensor) -> Tensor:
    """
    Converts tensor of angles to euclidean vectors.

    Arguments:
        - R: tensor of shape `(..., 1)` of angles.

    Returns:
        Tensor of shape `(..., 2)`.
    """
    vec = R.new_empty((*R.shape[:-1], 2))

    vec[..., 0] = torch.cos(R).squeeze(-1)
    vec[..., 1] = torch.cos(R).squeeze(-1)

    return vec


def uniform_grid(size: int, device: str | None = None) -> Tensor:
    """
    Returns a uniform grid of angles in the range `[0, 2pi)`.

    Arguments:
        - size: int denoting the number of elements in the grid.
        - device: optional str denoting the device.

    Returns:
        Tensor of shape (size, 1)
    """
    return torch.linspace(0, 2 * math.pi, size + 1, device=device)[:-1].view(-1, 1)


def random_grid(size: int, device: str | None = None) -> Tensor:
    """
    Returns a random grid of angles in the range `[0, 2pi)`.

    Arguments:
        - size: int denoting the number of elements in the grid.
        - device: optional str denoting the device.

    Returns:
        Tensor of shape (size, 1)
    """
    return 2 * math.pi * torch.rand(size, device=device).view(-1, 1)


def inverse_matrix(R: Tensor) -> Tensor:
    """
    Returns the inverse rotation matrices.

    Arguments:
        R: tensor of shape `(..., 2, 2)`.

    Returns:
        Tensor of shape `(..., 2, 2)` of inverted matrices.
    """
    return R.mT


def inverse_angle(R: Tensor) -> Tensor:
    """
    Returns the inverse rotation angles.

    Arguments:
        R: tensor of shape `(..., 1)`.

    Returns:
        Tensor of shape `(..., 1)` of inverted angles.
    """
    return -R


def left_apply_angle(R1: Tensor, R2: Tensor) -> Tensor:
    """
    Performs left group action of R1 on R2, parameterized
    as angles. The usual broadcasting rules apply.

    Arguments:
        - R1: tensor of shape `(..., 1)`.
        - R2: tensor of shape `(..., 1)`.

    Returns:
        Tensor containing left group action.
    """
    return (R1 + R2) % (2 * math.pi)


def left_apply_to_angle(R1: Tensor, R2: Tensor) -> Tensor:
    """
    Performs left group action of each element in R1
    to each element in R2, parameterized as angles.

    Arguments:
        - R1: tensor of shape `(N, 1)`.
        - R2: tensor of shape `(M, 1)`.

    Returns:
        Tensor of shape `(N, M, 1)`.
    """
    return (R1[:, None] + R2) % (2 * math.pi)


def left_apply_angle_to_R2(R: Tensor, grid: Tensor) -> Tensor:
    """
    Applies every rotation in R, parameterized as angles,
    to every vector in grid.

    Arguments:
        - R: tensor of shape `(N, 1)` of rotation angles.
        - grid: tensor of shape (W, H, 2) of vectors.

    Returns:
        Tensor of shape `(N, W, H, 2)`.
    """
    return left_apply_matrix_to_R2(angle_to_matrix(R), grid)


def left_apply_matrix_to_R2(R: Tensor, grid: Tensor) -> Tensor:
    """
    Applies every rotation in R, parameterized as matrices,
    to every vector in grid.

    Arguments:
        - R: tensor of shape `(N, 2, 2)` of rotation matrices.
        - grid: tensor of shape (W, H, 2) of vectors.

    Returns:
        Tensor of shape `(N, W, H, 2)`.
    """
    return (R[:, None, None] @ grid[..., None]).flatten(-2)


def geodesic_distance_euclid(R1: Tensor, R2: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Calculates the geodesic distance between R1 and R2, parameterized
    as euclidean vectors. Usual broadcasting rules apply.

    Arguments:
        - R1: tensor of shape (..., 2).
        - R2: tensor of shape (..., 2).

    Returns:
        Tensor containing the geodesic distance between R1 and R2.
    """
    return torch.acos(torch.clamp((R1 * R2).sum(-1), -1 + eps, 1 - eps))


def geodesic_distance(R1: Tensor, R2: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Calculates the geodesic distance between R1 and R2, parameterized
    as euclidean angles. Usual broadcasting rules apply.

    Arguments:
        - R1: tensor of shape (..., 1).
        - R2: tensor of shape (..., 1).

    Returns:
        Tensor containing the geodesic distance between R1 and R2.
    """
    return geodesic_distance_euclid(angle_to_euclid(R1), angle_to_euclid(R2), eps=eps)


def grid_sample(
    grid: Tensor,
    signal: Tensor,
    signal_grid: Tensor,
    mode: str = "rbf",
    width: float = 0.5,
) -> Tensor:
    """
    Samples signal for given grid of rotations, parameterized as angles,
    on given signal and corresponding signal grid.

    Arguments:
        - grid: tensor of shape (N, 1) of rotations to sample.
        - signal: tensor of shape (M, S) of signal to sample on.
        - signal_grid: tensor of shape (M, 1) of rotations on which signal
                       is defined.

    Returns:
        Tensor of shape (N, S) containing the sampled signal.
    """
    if mode == "rbf":
        return interpolation.interpolate_RBF(
            grid[None],
            signal_grid[None],
            signal[None],
            dist_fn=geodesic_distance,
            width=width,
        )[0]
    elif mode == "nearest":
        return interpolation.interpolate_NN(
            grid[None],
            signal_grid[None],
            signal[None],
            dist_fn=geodesic_distance,
        )[0]
    else:
        raise ValueError(f"Supported modes are 'rbf' or 'nearest', got {mode=}.")


def nearest_neighbour_distance(grid: Tensor) -> Tensor:
    """
    Returns the nearest neighbour distance for each element in grid.

    Arguments:
        grid: tensor of shape (N, 1) of rotation angles.
    """
    return (geodesic_distance(grid[:, None], grid)).min(-1)
