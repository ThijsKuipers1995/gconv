import torch
from torch import Tensor

import so3


def uniform_grid(size: tuple[int, int]) -> Tensor:
    n_rotations, n_reflections = size

    coeff1 = torch.ones(n_rotations, 1)
    coeff2 = -1 * torch.ones(n_reflections, 1)
    coeffs = torch.vstack((coeff1, coeff2))

    R1 = so3.uniform_grid(n_rotations, "matrix")
    R2 = so3.uniform_grid(n_reflections, "matrix")
    R = torch.vstack((R1, R2))

    grid = torch.cat((coeffs, R), dim=-1)

    return grid


def left_apply(H1: Tensor, H2: Tensor) -> Tensor:
    R1 = H1[:, 1:].view(-1, 3, 3)
    R2 = H2[:, 1:].view(-1, 3, 3)

    coeff1 = H1[:, 0].unsqueeze(-1)
    coeff2 = H2[:, 0].unsqueeze(-1)

    R = (R1[:, None] @ R2).flatten(-2, -1)
    coeff = coeff1[:, None] * coeff2

    return torch.cat((coeff, R), dim=-1)


def inverse(H: Tensor) -> Tensor:
    dims = H.shape[:-1]

    coeffs, R = H[..., 0, None], H[..., 1:].view(*dims, 3, 3)
    return torch.cat((coeffs, R.mT.flatten(-2, -1)), dim=-1)


def grid_sample(
    grid: Tensor,
    signal: Tensor,
    signal_grid: Tensor,
    signal_grid_size: tuple[int, int],
    rotation_mode: str = "rbf",
    reflection_mode: str = "rbf",
    rotation_width: float = 0.5,
    reflection_width: float = 0.5,
):
    n_rotations, n_reflections = signal_grid_size

    coeffs, R = grid[:, 0], grid[:, 1:].view(-1, 3, 3)

    so3_signal = signal[:n_rotations]
    so3_signal_grid = signal_grid[:n_rotations]

    r_signal = signal[n_rotations:]
    r_signal_grid = signal_grid[n_rotations:]

    so3_idx = torch.where(coeffs == 1)[0]
    r_idx = torch.where(coeffs == -1)[0]

    # sample rotations and reflections separately
    if n_rotations:
        so3_signal = so3.grid_sample(
            R[so3_idx],
            so3_signal,
            so3_signal_grid,
            mode=rotation_mode,
            width=rotation_width,
        )
    else:
        so3_signal = so3_idx

    if n_reflections:
        r_signal = so3.grid_sample(
            R[r_idx],
            r_signal,
            r_signal_grid,
            mode=reflection_mode,
            width=reflection_width,
        )
    else:
        r_signal = r_idx

    sampled_signal = torch.vstack((so3_signal, r_signal))

    # need to restore original order of O3 grid
    perms = torch.argsort(torch.stack((so3_idx, r_idx)))

    return sampled_signal[perms]
