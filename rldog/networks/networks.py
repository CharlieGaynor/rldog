import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Tuple

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
        ...


class BasePPONN(nn.Module, ABC):
    """
    boring template
    """

    def __init__(self) -> None:
        nn.Module.__init__(self)
        ABC.__init__(self)

    @abstractmethod
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @abstractmethod
    def forward_policy(self, state: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def forward_critic(self, state: torch.Tensor) -> torch.Tensor:
        ...


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


class BasicSoftMaxNN(BaseNN):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()

        self.l1 = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state: torch.Tensor) -> Any:
        output = self.l1(state)
        return self.softmax(output)


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


class StandardPPO(BasePPONN):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, hidden_layers: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        self.value_head = nn.Linear(hidden_size, 1)
        self.policy_head = nn.Linear(hidden_size, output_size)
        self.activation = nn.SiLU()
        self.sm = nn.Softmax(dim=-1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        for layer in self.layers:
            state = self.activation(layer(state))
        val = self.value_head(state)
        probs = self.sm(self.policy_head(state))
        return probs, val

    def forward_policy(self, state: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            state = self.activation(layer(state))
        probs = self.sm(self.policy_head(state))
        return probs

    def forward_critic(self, state: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            state = self.activation(layer(state))
        val = self.value_head(state)
        return val
