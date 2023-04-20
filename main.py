import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        tensors = (torch.zeros(3, 3), torch.zeros(3, 3))

        for i, tensor in enumerate(tensors):
            self.register_buffer(f"tensor{i}", tensor)

        self.tensors = (self.tensor0, self.tensor1)

    def forward(self):
        print(id(self.tensors[0]))
        print(id(self.tensor0))


def main():
    model = Model()
    model.to(device="mps")

    model()


if __name__ == "__main__":
    main()
