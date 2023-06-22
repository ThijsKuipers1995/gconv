from typing import Optional
import torch
from torch import Tensor

from gconv.geometry import so3

from gconv.geometry import repulsion

from math import pi


def spherical_to_euclid(g: Tensor) -> Tensor:
    """
    Converts spherical coordinates to euclidean coordinates.

    Arguments:
        - g: Tensor of shape `(..., 2)`.

    Returns:
        - Tensor of shape `(..., 3)`.
    """
    x = g.new_empty((*g.shape[:-1], 3))

    beta = g[..., 0]
    gamma = g[..., 1]

    x[..., 0] = torch.sin(beta) * torch.cos(gamma)
    x[..., 1] = torch.sin(beta) * torch.sin(gamma)
    x[..., 2] = torch.cos(beta)

    return x


def euclid_to_spherical(x: Tensor) -> Tensor:
    """
    Converts euclidean coordinates to spherical coordinates.

    Arguments:
        - g: Tensor of shape `(..., 3)`.

    Returns:
       -  Tensor of shape `(..., 2)`.
    """
    g = x.new_empty((*x.shape[:-1], 2))

    g[..., 0] = torch.acos(x[..., 2])
    g[..., 1] = torch.atan2(x[..., 1], x[..., 0])

    return g


def random_s2(shape: tuple[int, ...], device: Optional[str] = None) -> torch.Tensor:
    """
    Generates Tensor of uniformly sampled spherical coordinates on
    S2.

    Arguments:
        - shape: Shape of the output tensor.
        - device: Device on which the new tensor is created.

    Returns:
        - Tensor of shape (*shape, 3).
    """
    x = torch.randn((*shape, 3), device=device)
    return euclid_to_spherical(x / torch.linalg.norm(x, dim=-1, keepdim=True))


def geodesic_distance_s2(r1: Tensor, r2: Tensor, eps: float = 1e-7):
    return torch.acos(torch.clamp((r1 * r2).sum(-1), -1 + eps, 1 - eps))


def spherical_to_euler(g: Tensor) -> Tensor:
    alpha = g.new_zeros(g.shape[0], 1)
    return torch.hstack((alpha, g))


def spherical_to_euler_neg_gamma(g: Tensor) -> Tensor:
    minus_gamma = g[:, 1]
    return torch.hstack((minus_gamma, g))


def uniform_grid_s2(
    n: int,
    parameterization: str = "euclidean",
    set_alpha_as_neg_gamma: bool = False,
    steps: int = 100,
    step_size: float = 0.1,
    show_pbar: bool = True,
    device: Optional[str] = None,
) -> Tensor:
    """
    Creates a uniform grid of `n` rotations on S2. Rotations will be uniform
    with respect to the geodesic distance.

    Arguments:
        - n: int denoting the number of rotations in grid.
        - parameterization: Parameterization of the returned grid elements. Must
                            be either 'spherical', 'euclidean', 'quat', 'matrix', or 'euler'. Defaults to
                            'euclidean'.
        - steps: Number of minimization steps.
        - step_size: Strength of minimization step. Default of 0.1 works well.
        - show_pbar: If True, will show progress of optimization procedure.
        - device: Device on which energy minimization will be performed and on
                  which the output grid will be defined.

    Returns:
        - Tensor containing uniform grid on SO3.
    """
    add_alpha = False
    to_so3_fn = (
        spherical_to_euler_neg_gamma if set_alpha_as_neg_gamma else spherical_to_euler
    )

    match parameterization.lower():
        case "spherical":
            param_fn = lambda x: x
        case "euclidean":
            param_fn = spherical_to_euclid
        case "euler":
            add_alpha = True
            param_fn = lambda x: x
        case "matrix":
            add_alpha = True
            param_fn = so3.euler_to_matrix
        case "quat":
            add_alpha = True
            param_fn = so3.euler_to_quat

    grid = random_s2((n,), device=device)

    repulsion.repulse(
        grid,
        steps=steps,
        step_size=step_size,
        alpha=0.001,
        metric_fn=geodesic_distance_s2,
        transform_fn=spherical_to_euclid,
        dist_normalization_constant=pi,
        show_pbar=show_pbar,
        in_place=True,
    )

    grid = to_so3_fn(grid) if add_alpha else grid

    return param_fn(grid)
