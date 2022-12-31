import torch
import torch.nn as nn
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

class StandardNN(BaseNN):
    """
    Standard
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int, hidden_layers: int) -> None:
        super(StandardNN, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))

        # Define the ReLU activation function
        self.activation = nn.SiLU()

    def forward(self, state: torch.Tensor) -> torch.Tensor:

        for layer in self.layers[:-1]:
            state = self.activation(layer(state))
        return self.layers[-1](state)  # type: ignore[no-any-return]
    
class StandardSoftmaxNN(BaseNN):
    """
    Standard
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int, hidden_layers: int) -> None:
        super(StandardSoftmaxNN, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))

        # Define the ReLU activation function
        self.activation = nn.SiLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:

        for layer in self.layers[:-1]:
            state = self.activation(layer(state))
        return self.softmax(self.layers[-1](state))  # type: ignore[no-any-return]

