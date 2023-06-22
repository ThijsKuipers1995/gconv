import sys

sys.path.append("..")

import torch
from gconv.geometry.groups import o2

import gconv.nn.functional as gF


def test_uniform_grid():
    grid = o2.uniform_grid((4, 4))
    assert grid.shape == (8, 2)

    grid = o2.uniform_grid((4, 0))
    assert grid.shape == (4, 2)


def test_inverse():
    grid = o2.uniform_grid((4, 4))

    inverse_grid = o2.inverse(grid)
    assert torch.all(grid[:, 1] == -inverse_grid[:, 1])
    assert torch.all(grid == o2.inverse(inverse_grid))


def test_det():
    grid = o2.uniform_grid((4, 4))

    det = o2.det(grid)
    assert det.shape == (8, 1)
    assert torch.all(det == grid[:, 0][..., None])


def test_left_apply_o2():
    grid = o2.uniform_grid((4, 4))

    prod = o2.left_apply_angle(grid, grid)
    assert prod.shape == (8, 2)

    prod = o2.left_apply_angle(grid[:, None], grid)
    assert prod.shape == (8, 8, 2)


def test_left_apply_to_o2():
    grid = o2.uniform_grid((4, 4))

    prod = o2.left_apply_to_angle(grid, grid)
    assert prod.shape == (8, 8, 2)


def test_left_apply_to_Rn():
    grid = o2.uniform_grid((4, 4))

    grid_Rn = gF.create_grid_R2(5)

    prod = o2.left_apply_to_R2(grid, grid_Rn)
    assert prod.shape == (8, 5, 5, 2)


def test_grid_sample():
    grid = o2.uniform_grid((4, 4))
    signal = torch.rand(12, 64)
    signal_grid = o2.uniform_grid((6, 6))

    sampled_signal = o2.grid_sample(grid, signal, signal_grid, (6, 6))
    assert sampled_signal.shape == (8, 64)


def main():
    test_uniform_grid()
    test_inverse()
    test_det()
    test_left_apply_o2()
    test_left_apply_to_o2()
    test_left_apply_to_Rn()
    test_grid_sample()


if __name__ == "__main__":
    main()

    print("Tests finished succesfully!")
