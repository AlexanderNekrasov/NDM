import torch
from torch.utils.data import TensorDataset


def checkerboard_dataset(n=80000):
    X = torch.rand((8 * n, 2)) * 4
    X = X[(X[:, 0] // 1 + X[:, 1] // 1) % 2 == 0]
    X = X - X.mean()
    X = X / X.std()
    X = X[:n]
    return TensorDataset(X)


def get_dataset(name, n=80000):
    if name == "checkerboard":
        return checkerboard_dataset(n)
    else:
        raise ValueError(f"Unknown dataset: {name}")
