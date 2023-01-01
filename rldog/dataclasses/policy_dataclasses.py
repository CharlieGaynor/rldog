from dataclasses import dataclass

import gym
import torch
import torch.nn as nn


@dataclass
class reinforce_config:
    n_actions: int
    n_obs: int
    env: gym.Env

    clip_value: float
    obs_normalization_factor: float
    games_to_play: int
    one_hot_encode: bool
    input_size: int
    policy_network: nn.Module
    lr: float
    opt: torch.optim.Optimizer
    max_games: int
    gamma: float


@dataclass
class Transition:
    action_probs: torch.Tensor
    reward: float
