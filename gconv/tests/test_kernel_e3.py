import sys

sys.path.append("..")

from gconv.nn import kernels
from gconv.geometry.groups import o3


def test_lifting_kernel():
    grid_H = o3.uniform_grid((4, 4))

    kernel = kernels.GLiftingKernelE3(4, 6, 5)
    weight = kernel(grid_H)
    assert weight.shape == (6, 8, 4, 5, 5, 5)

    kernel = kernels.GLiftingKernelE3(4, 6, 5, groups=2)
    weight = kernel(grid_H)
    assert weight.shape == (6, 8, 2, 5, 5, 5)


def test_separable_kernel():
    grid_H = o3.uniform_grid((4, 4))

    kernel = kernels.GSeparableKernelE3(4, 6, 5, (4, 4))
    weight_H, weight_Rn = kernel(grid_H, grid_H)
    assert weight_H.shape == (6, 8, 4, 8, 1, 1, 1)
    assert weight_Rn.shape == (6, 8, 1, 5, 5, 5)

    kernel = kernels.GSeparableKernelE3(4, 6, 5, 8, groups=2)
    weight_H, weight_Rn = kernel(grid_H, grid_H)
    assert weight_H.shape == (6, 8, 2, 8, 1, 1, 1)
    assert weight_Rn.shape == (6, 8, 1, 5, 5, 5)


def test_group_kernel():
    grid_H = o3.uniform_grid((4, 4))

    kernel = kernels.GKernelE3(4, 6, 5, 8)
    weight = kernel(grid_H, grid_H)
    assert weight.shape == (6, 8, 4, 8, 5, 5, 5)

    kernel = kernels.GKernelE3(4, 6, 5, 8, groups=2)
    weight = kernel(grid_H, grid_H)
    assert weight.shape == (6, 8, 2, 8, 5, 5, 5)


def main():
    test_lifting_kernel()
    test_separable_kernel()
    test_group_kernel()


if __name__ == "__main__":
    main()

    print("Finished tests succesfully!")
