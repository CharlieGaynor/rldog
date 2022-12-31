import gym
import torch
import torch.nn as nn

from rldog.configs.base_config import BaseConfig
from rldog.networks.networks import BasicNN


class FrozenLakeConfig(BaseConfig):
    def __init__(self, is_slippery: bool = False):
        self.n_actions = 4
        self.n_obs = 16
        self.state_type = "DISCRETE"
        self.env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery, disable_env_checker=True)
        self.policy_network: nn.Module = BasicNN(input_size=self.n_obs, output_size=self.n_actions)

    def DQN_config(
        self,
        network: nn.Module = None,
        games_to_play: int = 1000,
        one_hot_encode: bool = True,
        lr: float = 1e-3,
        alpha: float = 0.1,
        gamma: float = 0.99,
        mini_batch_size: int = 4,
        buffer_size: int = 128,
        min_epsilon: float = 0.2,
        initial_epsilon: float = 1,
        epsilon_grace_period: float = 0.5,
        obs_normalization_factor: float = 1,
    ) -> None:
        if network is not None:
            self.policy_network = network

        self.games_to_play = games_to_play
        self.one_hot_encode = one_hot_encode
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma
        self.mini_batch_size = mini_batch_size
        self.buffer_size = buffer_size
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.obs_normalization_factor = obs_normalization_factor

        self.opt = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.epsilon_grace_period: int = int(self.games_to_play * epsilon_grace_period)
        self.games_to_decay_epsilon_for: int = (self.games_to_play - self.epsilon_grace_period) * 3 // 4
