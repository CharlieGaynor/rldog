import torch
import torch.nn as nn

from rldog.networks.base_nn import BaseNN


class StandardNN(BaseNN):
    """
    very basic NN
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
