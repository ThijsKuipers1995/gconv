import plotly.graph_objects as go
from math import pi
import torch


def plot_signal(
    n: int,
    signal,  # n x n
    alpha_steps: float = 1,
) -> None:
    """ """

    # set up figure layouts
    _axis = dict(
        showbackground=False,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        title="",
        nticks=3,
    )

    _layout = dict(
        scene=dict(
            xaxis=dict(
                **_axis,
            ),
            yaxis=dict(
                **_axis,
            ),
            zaxis=dict(
                **_axis,
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
    )

    _cmap_bwr = [[0, "rgb(0,50,255)"], [0.5, "rgb(200,200,200)"], [1, "rgb(255,50,0)"]]

    # generate spherical mesh
    beta = torch.linspace(0, pi, n)
    gamma = torch.linspace(0, 2 * pi, n)

    x = torch.outer(torch.cos(gamma), torch.sin(beta))
    y = torch.outer(torch.sin(gamma), torch.sin(beta))
    z = torch.outer(torch.ones(n), torch.cos(beta))

    # s2_grid = R.euclid_to_spherical(
    #     torch.vstack((x.flatten(), y.flatten(), z.flatten())).T
    # )

    frames = []
    # generate frames, interpolating for each alpha
    for alpha in torch.linspace(-pi, pi, alpha_steps):
        # sphere_mesh = R.euler_to_quat(
        #     torch.hstack((alpha * torch.ones(n * n, 1), s2_grid)).view(-1, 3)
        # )

        frames.append(
            go.Frame(
                data=go.Surface(
                    x=x,
                    y=y,
                    z=z,
                    surfacecolor=signal,
                    colorscale=_cmap_bwr,
                    cmin=-1,
                    cmax=1,
                ),
                name=f"{alpha.item():.02f}",
            )
        )

    # To show surface at figure initialization
    fig = go.Figure(layout=_layout, frames=frames)
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=signal,
            colorscale=_cmap_bwr,
            cmin=-1,
            cmax=1,
        )
    )

    # for sliding trough alpha slices
    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [f.name],
                        {
                            "frame": {"duration": 0},
                            "mode": "immediate",
                            "fromcurrent": True,
                            "transition": {"duration": 0, "easing": "linear"},
                        },
                    ],
                    "label": str(f.name),
                    "method": "animate",
                }
                for f in fig.frames
            ],
        }
    ]

    fig.update_layout(sliders=sliders)
    fig.show()


if __name__ == "__main__":
    n = 50
    signal = torch.randn(n, n)

    plot_signal(n, signal)
