from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseNN(nn.Module, ABC):
    """
    boring template
    """

    def __init__(self) -> None:
        nn.Module.__init__(self)
        ABC.__init__(self)

    @abstractmethod
    def forward(self, state: torch.Tensor) -> Any:
        pass
