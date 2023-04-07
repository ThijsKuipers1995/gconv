# 3D Group Convolutions

This package implements a framework for 3D group convolutions that are easy to use and implement in existing Pytorch modules. The package offers premade modules for SE3 convolutions, as well as basic **operations** such as pooling and normalization for $R^n \rtimes H$ input.

## Getting Started

The gconv3d modules are as straightforward to use as any other regular convolution module from python, as most group related mechanisms are abstracted away. The group mechanisms such as interpolation or group actions are fully accessible and custamizable, and custom group convolution kernels and modules are easily implemented (for this, see `gconv_tutorial.ipynb`).

```python3
from gconv import gnn                                                               # 1
from gconv.geometry import rotation as R                                            # 2
import torch                                                                        # 3
                                                                                    # 4
x1 = torch.randn(16, 3, 28, 28, 28)                                                 # 5
H = R.uniform_grid(12)                                                              # 6
                                                                                    # 7
lifting_layer = gnn.GConvLiftingSE3(in_channels=3, out_channels=16, kernel_size=5)  # 8
gconv_layer = gnn.GConvSE3(in_channels=16, out_channels=32, kernel_size=5)          # 9
                                                                                    # 10
pool_h = gnn.AvgPoolH()                                                             # 11                                    
                                                                                    # 12
x2, H = lifting_layer(x1, H)                                                        # 13
x3, H = gconv_layer(x2, H)                                                          # 14
y = pool_h(x3)                                                                      # 15
```

In line 5 we create a random batch of three-channel $R^3$ volumes and in line 6 and we create a uniform grid consisting of 12 $\text{SO}(3)$ elements to which we want to lift the volumes. In line 13 the input defined on $R^3$ is lifted to $R^3 \rtimes \text{SO}(3) = \text{SE}(3)$. In line 14 an $\text{SE}(3)$ convolution is performed. In line 15 the $\text{SE}(3)$ activations are reduced back to $R^3$.

## Requirements:
```
python >= 3.10
pytorch
numpy
matplotlib
plotly
```
For running the training scripts:
```
medmnist
deeprank
wandb
```
