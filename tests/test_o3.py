import sys

sys.path.append("..")

import torch
from gconv.geometry.groups import o3

import gconv.nn.functional as gF


def test_uniform_grid():
    grid = o3.uniform_grid((4, 4))
    assert grid.shape == (8, 10)

    grid = o3.uniform_grid((4, 0))
    assert grid.shape == (4, 10)


def test_inverse():
    grid = o3.uniform_grid((4, 4))

    inverse_grid = o3.inverse(grid)
    assert inverse_grid.shape == (8, 10)
    assert torch.all(
        grid[:, 1:].view(-1, 3, 3) == inverse_grid[:, 1:].view(-1, 3, 3).mT
    )


def test_det():
    grid = o3.uniform_grid((4, 4))

    det = o3.det(grid)
    assert det.shape == (8, 1)
    assert torch.all(det == grid[:, 0][..., None])


def test_left_apply_o3():
    grid = o3.uniform_grid((4, 4))

    prod = o3.left_apply_O3(grid, grid)
    assert prod.shape == (8, 10)

    prod = o3.left_apply_O3(grid[:, None], grid)
    assert prod.shape == (8, 8, 10)


def test_left_apply_to_o3():
    grid = o3.uniform_grid((4, 4))

    prod = o3.left_apply_to_O3(grid, grid)
    assert prod.shape == (8, 8, 10)


def test_left_apply_to_Rn():
    grid = o3.uniform_grid((4, 4))

    grid_Rn = gF.create_grid_R3(5)

    prod = o3.left_apply_to_R3(grid, grid_Rn)
    assert prod.shape == (8, 5, 5, 5, 3)


def test_grid_sample():
    grid = o3.uniform_grid((4, 4))
    signal = torch.rand(12, 64)
    signal_grid = o3.uniform_grid((6, 6))

    sampled_signal = o3.grid_sample(grid, signal, signal_grid, (6, 6))
    assert sampled_signal.shape == (8, 64)


def main():
    test_uniform_grid()
    test_inverse()
    test_det()
    test_left_apply_o3()
    test_left_apply_to_o3()
    test_left_apply_to_Rn()
    test_grid_sample()


if __name__ == "__main__":
    main()

    print("Tests finished succesfully!")
