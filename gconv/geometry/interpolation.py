from typing import Callable
import torch
from torch import Tensor


def RBF_gauss(x: Tensor, width: float) -> Tensor:
    """
    Gaussian radial basis function.
    ln2 = 0.69314718

    Arguments:
        - x: Input tensor.
        - width: Float denoting the half width at half height of
                 the gaussian function.

    Returns:
        - Tensor of RBF values.
    """
    return torch.exp(-(x**2 / (width**2 / 0.69314718)))


def interpolate_RBF(
    nodes: Tensor,
    grid: Tensor,
    signal: Tensor,
    *,
    dist_fn: Callable[[Tensor], Tensor],
    rbf: Callable[[Tensor], Tensor] = RBF_gauss,
    width: float = 2.0,
) -> Tensor:
    """
    Performs radial basis function interpolation of rotations.

    Arguments:
        Arguments:
        - nodes: Tensor of shape `(N, K, D)` of nodes to interpolate.
        - grid: Tensor of shape `(N, L, D)` denoting the discrete
                grid of nodes on which to interpolate.
        - signal: Tensor of shape `(N, L, S)` containing the signal
                  corresponding to `grid`.
        - dist_fn: Distance function used to calculate distance between
                   nodes.
        - rbf: Radial basis function used for interpolation, defaults
               to gaussian RBF.
        - width: Width of the RBF.

    Returns:
        - Tensor containing the interpolant of shape '(N, K, S)'.
    """
    m = rbf(dist_fn(grid[..., None, :], grid[..., None, :, :]), width)

    coeffs = torch.linalg.solve(m, signal).transpose(-1, -2)
    p = rbf(dist_fn(nodes[..., None, :], grid[..., None, :, :]), width)

    return p @ coeffs.transpose(-1, -2)


def interpolate_NN(
    nodes: Tensor,
    grid: Tensor,
    signal: Tensor,
    *,
    dist_fn: Callable = Callable[[Tensor], Tensor],
) -> Tensor:
    """
    Performs nearest neighbour interpolation of nodes with
    respect to `signal` that is defined on `grid`.

    Arguments:
        - nodes: Tensor of shape `(N, K, D)` of nodes to interpolate.
        - grid: Tensor of shape `(N, L, D)` denoting the discrete
                grid of nodes on which to interpolate.
        - signal: Tensor of shape `(N, L, S)` containing the signal
                  corresponding to `grid`.
        - dist_fn: Distance function used to calculate distance between
                   nodes.

    Returns:
        - Tensor of shape `(N, K, S)`.
    """
    dims = signal.shape[2:]

    dists = dist_fn(nodes[..., :, None, :], grid[..., None, :, :])
    idx = torch.topk(dists, k=1, largest=False)[1]

    return signal.gather(1, idx.expand(-1, -1, *dims))


def interpolate_BCC(
    nodes: Tensor,
    grid: Tensor,
    signal: Tensor,
    *,
    dist_fn=Callable[[Tensor], Tensor],
    eps: float = 1e-6,
) -> Tensor:
    """
    Performs barycentric coordinates interpolation.

    Arguments:
        - nodes: Tensor of shape `(N, K, D)` of nodes to interpolate.
        - grid: Tensor of shape `(N, L, D)` denoting the discrete
                grid of nodes on which to interpolate.
        - signal: Tensor of shape `(N, L, S)` containing the signal
                  corresponding to `grid`.
        - dist_fn: Distance function used to calculate distance between
                   nodes.
        - eps: Float for preventing numerical instabilities.

    Returns:
        - Tensor of shape `(N, K, S)`.
    """
    N, _, S = signal.shape
    _, H, D = nodes.shape

    dists = dist_fn(nodes[..., None, :], grid[..., None, :, :])
    dists_k, idx = torch.topk(dists, k=3, largest=False)

    simplices = (
        grid[..., None, :]
        .expand(-1, -1, D, -1)
        .gather(1, idx[..., None].expand(-1, -1, -1, D))
    )

    bcc = torch.linalg.lstsq(simplices.transpose(-1, -2), nodes[..., None])[0].view(
        -1, 3
    )

    mask = dists_k[:, :, 0].view(-1) <= eps
    bcc[mask, 0], bcc[mask, 1], bcc[mask, 2] = 1.0, 0.0, 0.0
    bcc[bcc < 0] = 0

    bcc = bcc.view(N, H, 3)
    bcc /= bcc.sum(-1, keepdim=True)

    signal = (
        signal[..., None, :]
        .expand(-1, -1, 3, -1)
        .gather(1, idx[..., None].expand(-1, -1, -1, S))
    )

    return torch.sum(bcc[..., None] * signal, axis=-2)
