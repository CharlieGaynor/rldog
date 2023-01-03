import gym
import torch
import torch.nn as nn

from rldog.configs.base_config import BaseConfig
from rldog.networks.networks import StandardPPO


class CartPoleConfig(BaseConfig):
    def __init__(self) -> None:
        self.n_actions = 2
        self.n_obs = 4
        self.env = gym.make("CartPole-v1")

    def PPO_config(
        self,
        net: nn.Module = None,
        old_net: nn.Module = None,
        games_to_play: int = 5000,
        one_hot_encode: bool = False,
        gamma: float = 0.99,
        lr: float = 0.003,
        obs_normalization_factor: float = 1,
        clip_value: float = 0.5,
        n_games_per_learning_batch: int = 100,
        n_learning_episodes_per_batch: int = 100,
        use_parallel: bool = False,
        n_envs: int = 1,
    ) -> None:

        if net is None:
            self.net = StandardPPO(input_size=4, output_size=2, hidden_layers=2, hidden_size=64)
            self.old_net = StandardPPO(input_size=4, output_size=2, hidden_layers=2, hidden_size=64)
        else:
            self.net = net
            self.old_net = old_net  # type: ignore[assignment]

        self.lr = lr

        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.gamma = gamma
        self.games_to_play = games_to_play
        self.n_games_per_learning_batch = n_games_per_learning_batch
        self.n_learning_episodes_per_batch = n_learning_episodes_per_batch
        self.one_hot_encode = one_hot_encode

        self.use_parallel = use_parallel
        self.n_envs = n_envs

        self.obs_normalization_factor = obs_normalization_factor
        self.clip_value = clip_value
