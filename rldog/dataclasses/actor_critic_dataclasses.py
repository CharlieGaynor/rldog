from dataclasses import dataclass

import gym
import torch
import torch.nn as nn


@dataclass
class ActorCriticConfig:
    n_actions: int
    n_obs: int
    env: gym.Env

    actor: nn.Module
    critic: nn.Module
    lr: float
    actor_opt: torch.optim.Optimizer
    critic_opt: torch.optim.Optimizer

    gamma: float
    games_to_play: int
    one_hot_encode: bool
    obs_normalization_factor: float
    clip_value: float


@dataclass
class Transition:
    action_probs: torch.Tensor
    reward: float
    critic_value: torch.FloatTensor
