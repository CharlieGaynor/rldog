from dataclasses import dataclass

import gym
import torch
import torch.nn as nn
from rldog.networks.networks import BasePPONN


@dataclass
class ReinforceConfig:
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
class PPOConfig:
    n_actions: int
    n_obs: int
    env: gym.Env

    net: BasePPONN
    old_net: BasePPONN
    lr: float
    actor_opt: torch.optim.Optimizer
    critic_opt: torch.optim.Optimizer

    gamma: float
    games_to_play: int

    n_games_per_learning_batch: int  # For each learning batch, play this many games
    n_learning_episodes_per_batch: int  # How many times do we want to learn from the games played?

    one_hot_encode: bool
    obs_normalization_factor: float
    clip_value: float

    use_parallel: bool


@dataclass
class Transition:
    action_probs: torch.Tensor
    reward: float


@dataclass
class PPOTransition:
    obs: torch.Tensor
    action: int
    reward: float
