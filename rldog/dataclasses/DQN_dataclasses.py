from dataclasses import dataclass
from typing import Type

import gym
import torch
import torch.nn as nn

from rldog.networks.base_nn import BaseNN


@dataclass
class DQN_config:
    n_actions: int
    n_obs: int
    state_type: str
    name: str
    unit_price: float
    env: gym.Env

    policy_network: nn.Module
    lr: float
    opt: torch.optim.Optimizer
    max_games: int
    epsilon: float
    min_epsilon: float
    games_to_decay_epsilon_for: int
    alpha: float
    gamma: float
    mini_batch_size: int
    buffer_size: int
