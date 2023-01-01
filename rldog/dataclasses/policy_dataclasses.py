from dataclasses import dataclass

import gym
import torch
import torch.nn as nn


@dataclass
class DQN_config:
    n_actions: int
    n_obs: int
    state_type: str
    name: str
    unit_price: float
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
    epsilon: float
    min_epsilon: float
    games_to_decay_epsilon_for: int
    epsilon_grace_period: int
    alpha: float
    gamma: float
    mini_batch_size: int
    buffer_size: int


@dataclass
class Transition:
    action_probs: torch.Tensor
    reward: float
