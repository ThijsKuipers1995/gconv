import sys

sys.path.append("..")

import torch

from gconv.nn import GLiftingConvSE2, GSeparableConvSE2, GConvSE2
from gconv.geometry.groups import so2


import warnings

warnings.filterwarnings("ignore")


def test_lifting_gconv():
    batch_size = 1
    in_channels = 1
    out_channels = 2
    kernel_size = 3
    group_kernel_size = 4
    groups = 1
    bias = True

    input = torch.rand(batch_size, in_channels, 28, 28)
    grid_H = so2.uniform_grid(8)

    model = GLiftingConvSE2(
        in_channels,
        out_channels,
        kernel_size,
        group_kernel_size=group_kernel_size,
        padding="same",
        groups=groups,
        bias=bias,
    )

    output, out_H = model(input)

    assert output.shape == (batch_size, out_channels, group_kernel_size, 28, 28)

    model = GLiftingConvSE2(
        in_channels,
        out_channels,
        kernel_size,
        group_kernel_size=group_kernel_size,
        padding="same",
        groups=groups,
        bias=bias,
        grid_H=grid_H,
        permute_output_grid=False,
        sampling_mode="nearest",
        mask=True,
    )

    output, out_H = model(input)

    assert torch.allclose(out_H, grid_H)


def test_separable_gconv():
    batch_size = 3
    in_channels = 2
    out_channels = 6
    kernel_size = 3
    group_kernel_size = 8
    groups = 1
    bias = True

    input = torch.rand(batch_size, in_channels, group_kernel_size, 28, 28)
    grid_H = so2.uniform_grid(8)

    model = GSeparableConvSE2(
        in_channels,
        out_channels,
        kernel_size,
        group_kernel_size=group_kernel_size,
        padding="same",
        groups=groups,
        bias=bias,
    )

    output, out_H = model(input, grid_H)

    assert output.shape == (batch_size, out_channels, group_kernel_size, 28, 28)

    model = GSeparableConvSE2(
        in_channels,
        out_channels,
        kernel_size,
        group_kernel_size=group_kernel_size,
        padding="same",
        groups=groups,
        bias=bias,
        grid_H=grid_H,
        permute_output_grid=False,
    )

    output, out_H = model(input, grid_H)

    assert torch.allclose(out_H, grid_H)


def test_gconv():
    batch_size = 3
    in_channels = 2
    out_channels = 6
    kernel_size = 3
    group_kernel_size = 8
    groups = 1
    bias = True

    input = torch.rand(batch_size, in_channels, group_kernel_size, 28, 28)
    grid_H = so2.uniform_grid(8)

    model = GConvSE2(
        in_channels,
        out_channels,
        kernel_size,
        group_kernel_size=group_kernel_size,
        padding="same",
        groups=groups,
        bias=bias,
    )

    output, out_H = model(input, grid_H)

    assert output.shape == (batch_size, out_channels, group_kernel_size, 28, 28)

    model = GConvSE2(
        in_channels,
        out_channels,
        kernel_size,
        group_kernel_size=group_kernel_size,
        padding="same",
        groups=groups,
        bias=bias,
        grid_H=grid_H,
        permute_output_grid=False,
    )

    output, out_H = model(input, grid_H)

    assert torch.allclose(out_H, grid_H)


def main():
    test_lifting_gconv()
    test_separable_gconv()
    test_gconv()


if __name__ == "__main__":
    main()

    print("Finished tests succesfully!")
