# Continuous Regular Group Convolutions (WIP 👷‍♀️👷‍♂️)

This package implements a Pytorch framework for group convolutions that are easy to use and implement in existing Pytorch modules. The package offers premade modules for E3 and SE3 convolutions, as well as basic operations such as pooling and normalization for $\mathbb{R}^n \rtimes H$ input.

## Getting Started

The `gconv` modules are as straightforward to use as any regular Pytorch convolution module. The only difference is the output consisting of both the feature maps, as well as the group elements on which they are defined. See the example below:

```python3
import torch                                                                        # 1
import gconv.nn as gnn                                                              # 2
                                                                                    # 3
x1 = torch.randn(16, 3, 28, 28, 28)                                                 # 4
                                                                                    # 5
lifting_layer = gnn.GLiftingConvSE3(in_channels=3, out_channels=16, kernel_size=5)  # 6
gconv_layer = gnn.GSeparableConvSE3(in_channels=16, out_channels=32, kernel_size=5) # 7
                                                                                    # 8
pool = gnn.GAvgGlobalPool()                                                         # 9
                                                                                    # 10
x2, H1 = lifting_layer(x1)                                                          # 11
x3, H2 = gconv_layer(x2, H1)                                                        # 12
                                                                                    # 13
y = pool(x3, H2)                                                                    # 14
```

In line 5, a random batch of three-channel $\mathbb{R}^3$ volumes is created. In line 6, the $\mathbb{R}^3$ is lifted to $\text{SE}(3) = \mathbb{R}^3 \rtimes \text{SO}(3)$.  In line 7, an $\text{SE}(3)$ convolution is performed. In line 14, a global pooling is performed, resulting in $\text{SE}(3)$ invariant features.

Furthermore, `gconv` offers all the necessary tools to build fully custom group convolutions. All that is required is implementing 5 (or less, depending on the type of convolution) group ops! For more details on how to implement custom group convolutions, see `gconv_tutorial.ipynb`.

## Requirements:
```
python >= 3.10
torch
tqdm
```

## Reference:
Paper accepted at MICCAI 2023.
```
@misc{kuipers2023regular,
      title={Regular SE(3) Group Convolutions for Volumetric Medical Image Analysis}, 
      author={Thijs P. Kuipers and Erik J. Bekkers},
      year={2023},
      eprint={2306.13960},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
