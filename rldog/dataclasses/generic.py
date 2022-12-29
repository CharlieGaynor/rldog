from dataclasses import dataclass

import torch


@dataclass
class Transition:
    obs: torch.Tensor
    action: int
    reward: float
    next_obs: torch.Tensor
    terminated: bool
