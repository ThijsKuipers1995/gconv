import sys

sys.path.append("..")


import torch

from gconv.nn import GLiftingConvE3, GSeparableConvE3, GConvE3
from gconv.geometry.groups import o3


import warnings

warnings.filterwarnings("ignore")


def test_lifting_gconv():
    batch_size = 1
    in_channels = 1
    out_channels = 2
    kernel_size = 3
    group_kernel_size = 8
    groups = 1
    bias = True

    input = torch.rand(batch_size, in_channels, 28, 28, 28)
    grid_H = o3.uniform_grid((4, 4))

    model = GLiftingConvE3(
        in_channels,
        out_channels,
        kernel_size,
        group_kernel_size=(4, 4),
        padding="same",
        groups=groups,
        bias=bias,
    )

    output, out_H = model(input)

    assert output.shape == (batch_size, out_channels, group_kernel_size, 28, 28, 28)

    model = GLiftingConvE3(
        in_channels,
        out_channels,
        kernel_size,
        group_kernel_size=(4, 4),
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

    input = torch.rand(batch_size, in_channels, group_kernel_size, 28, 28, 28)
    grid_H = o3.uniform_grid((4, 4))

    model = GSeparableConvE3(
        in_channels,
        out_channels,
        kernel_size,
        group_kernel_size=(4, 4),
        padding="same",
        groups=groups,
        bias=bias,
    )

    output, out_H = model(input, grid_H)

    assert output.shape == (batch_size, out_channels, group_kernel_size, 28, 28, 28)

    model = GSeparableConvE3(
        in_channels,
        out_channels,
        kernel_size,
        group_kernel_size=(4, 4),
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

    input = torch.rand(batch_size, in_channels, group_kernel_size, 28, 28, 28)
    grid_H = o3.uniform_grid((4, 4))

    model = GConvE3(
        in_channels,
        out_channels,
        kernel_size,
        group_kernel_size=(4, 4),
        padding="same",
        groups=groups,
        bias=bias,
    )

    output, out_H = model(input, grid_H)

    assert output.shape == (batch_size, out_channels, group_kernel_size, 28, 28, 28)

    model = GConvE3(
        in_channels,
        out_channels,
        kernel_size,
        group_kernel_size=(4, 4),
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
