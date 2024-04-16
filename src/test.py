from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Callable
from torch import nn


def test_classifier(test_loader: DataLoader,
                    model: nn.Module, optimizer: Optimizer, loss: Callable, epochs: int,
                    save_dir: str):
    pass
