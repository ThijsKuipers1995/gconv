import sys

sys.path.append("..")

import math
import torch

from gconv.geometry import so2
from gconv.nn.functional import create_grid_R2


def test_uniform_grid():
    grid = so2.uniform_grid(8)
    reference = torch.Tensor(
        [[0.0000], [0.7854], [1.5708], [2.3562], [3.1416], [3.9270], [4.7124], [5.4978]]
    )

    assert torch.allclose(grid, reference)


def test_random_grid():
    grid = so2.random_grid(100)

    assert torch.all((grid >= 0) & (grid < 2 * math.pi))


def test_angle_to_matrix():
    grid = so2.random_grid(100)
    matrices = so2.angle_to_matrix(grid)

    assert torch.allclose(torch.det(matrices), torch.ones_like(grid))


def test_angle_to_euclid():
    grid = so2.random_grid(100)
    vectors = so2.angle_to_euclid(grid)

    assert torch.allclose(torch.norm(vectors, dim=-1), torch.ones_like(grid))


def test_geodesic_distance():
    grid = so2.random_grid(100)

    assert torch.all(so2.geodesic_distance(grid, grid) < 0.001)


def test_grid_sample():
    grid = so2.random_grid(100)

    signal = torch.rand(8, 64)
    signal_grid = so2.uniform_grid(8)

    new_signal = so2.grid_sample(grid, signal, signal_grid)

    assert new_signal.shape == (100, 64)


def main():
    test_uniform_grid()
    test_random_grid()
    test_angle_to_matrix()
    test_angle_to_euclid()
    test_geodesic_distance()
    test_grid_sample()


if __name__ == "__main__":
    main()
