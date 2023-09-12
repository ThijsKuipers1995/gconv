import sys

sys.path.append("..")


from gconv.gnn import GLiftingConvSE3
from gconv.geometry.groups import so3 as R
from gconv.gnn import functional as gF
import torch

from matplotlib import pyplot as plt

from torch.nn.functional import grid_sample


def plot_activations(activations):
    B, _, H, *_ = activations.shape
    fig = plt.figure()
    for i in range(B):
        for j in range(H):
            ax = fig.add_subplot(B, H, 1 + j + i * H)
            ax.imshow(activations[i, 1, j, 2].detach().numpy())
            ax.axis(False)
    plt.show()


def test_se3_lifting_conv():
    torch.manual_seed(0)

    batch_size = 1
    in_channels = 2
    out_channels = 3
    kernel_size = 5
    group_kernel_size = 4
    groups = 1
    bias = False

    input = torch.zeros(batch_size, in_channels, 5, 5, 5)
    input[:, :, 2, 2, :] = 1

    grid_H = torch.Tensor(
        [
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            [
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1],
            ],
        ]
    )
    from math import pi

    grid_H = R.matrix_z(torch.linspace(0, 2 * pi, 5)[:-1])

    grid_R3 = gF.create_grid_R3(5)

    grid_R3_rotated = R.left_apply_to_R3(grid_H, grid_R3)
    input_rotated = grid_sample(
        input.repeat(grid_H.shape[0], 1, 1, 1, 1),
        grid_R3_rotated,
        mode="nearest",
        padding_mode="zeros",
    )

    model = GLiftingConvSE3(
        in_channels,
        out_channels,
        kernel_size,
        group_kernel_size=group_kernel_size,
        padding="same",
        groups=groups,
        bias=bias,
        sampling_mode="nearest",
        sampling_padding_mode="zeros",
        mask=True,
        permute_output_grid=False,
    )

    output, H = model(input_rotated, grid_H)

    # plot_activations(input[:, :, None])
    # plot_activations(input_rotated[:, :, None])
    print(output.shape)
    plot_activations(output)


def main():
    test_se3_lifting_conv()


if __name__ == "__main__":
    main()
