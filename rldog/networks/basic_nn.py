from typing import Any

import torch
import torch.nn as nn

from rldog.networks.base_nn import BaseNN


class BasicNN(BaseNN):
    """
    very basic NN
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        super(BasicNN, self).__init__()

        self.l1 = nn.Linear(input_size, output_size)

    def forward(self, state: torch.Tensor) -> Any:
        output = self.l1(state)
        return output
