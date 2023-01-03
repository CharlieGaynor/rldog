from dataclasses import dataclass

import gym
import torch
import torch.nn as nn


@dataclass
class GenericConfig:
    n_actions: int
    n_obs: int
    state_type: str
    name: str
    unit_price: float
    env: gym.Env

    obs_normalization_factor: float
    games_to_play: int
    one_hot_encode: bool
    input_size: int
    policy_network: nn.Module
    lr: float
    opt: torch.optim.Optimizer
    max_games: int
    mini_batch_size: int
    buffer_size: int


@dataclass
class Transition:
    obs: torch.Tensor
    action: int
    reward: float
    next_obs: torch.Tensor
    terminated: bool
