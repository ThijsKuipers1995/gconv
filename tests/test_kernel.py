from gconv.nn import kernels
from gconv.geometry.groups import so3


def test_lifting_kernel():
    grid_H = so3.uniform_grid(3, "matrix")

    kernel = kernels.GLiftingKernelSE3(4, 6, 5)
    weight = kernel(grid_H)
    assert weight.shape == (6, 3, 4, 5, 5, 5)

    kernel = kernels.GLiftingKernelSE3(4, 6, 5, groups=2)
    weight = kernel(grid_H)
    assert weight.shape == (6, 3, 2, 5, 5, 5)


def test_separable_kernel():
    grid_H = so3.uniform_grid(3, "matrix")

    kernel = kernels.GSeparableKernelSE3(4, 6, 5, 3)
    weight_H, weight_Rn = kernel(grid_H, grid_H)
    assert weight_H.shape == (6, 3, 4, 3, 1, 1, 1)
    assert weight_Rn.shape == (6, 3, 1, 5, 5, 5)

    kernel = kernels.GSeparableKernelSE3(4, 6, 5, 3, groups=2)
    weight_H, weight_Rn = kernel(grid_H, grid_H)
    assert weight_H.shape == (6, 3, 2, 3, 1, 1, 1)
    assert weight_Rn.shape == (6, 3, 1, 5, 5, 5)


def test_subgroup_kernel():
    grid_H = so3.uniform_grid(3, "matrix")

    kernel = kernels.GSubgroupKernelSO3(4, 6, 5)
    weight = kernel(grid_H, grid_H)
    assert weight.shape == (6, 3, 4, 3, 1, 1, 1)

    kernel = kernels.GSubgroupKernelSO3(4, 6, 5, groups=2)
    weight = kernel(grid_H, grid_H)
    assert weight.shape == (6, 3, 2, 3, 1, 1, 1)


def test_group_kernel():
    grid_H = so3.uniform_grid(3, "matrix")

    kernel = kernels.GKernelSE3(4, 6, 5, 3)
    weight = kernel(grid_H, grid_H)
    assert weight.shape == (6, 3, 4, 3, 5, 5, 5)

    kernel = kernels.GKernelSE3(4, 6, 5, 3, groups=2)
    weight = kernel(grid_H, grid_H)
    assert weight.shape == (6, 3, 2, 3, 5, 5, 5)


def main():
    test_lifting_kernel()
    test_separable_kernel()
    test_subgroup_kernel()
    test_group_kernel()


if __name__ == "__main__":
    main()
