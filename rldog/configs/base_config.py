from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from rldog.networks.networks import BasicNN


class BaseConfig(ABC):
    @abstractmethod
    def __init__(self, max_games: int) -> None:
        self.n_obs: int
        self.n_actions: int

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
            self.policy_network = BasicNN(input_size=self.n_obs, output_size=self.n_actions)

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

    def reinforce_config(
        self,
        network: nn.Module = None,
        games_to_play: int = 1000,
        one_hot_encode: bool = True,
        gamma: float = 0.99,
        lr: float = 1e-3,
        obs_normalization_factor: float = 1,
        clip_value: float = 1,
    ) -> None:
        if network is not None:
            self.policy_network = BasicNN(input_size=self.n_obs, output_size=self.n_actions)

        self.gamma = gamma
        self.games_to_play = games_to_play
        self.one_hot_encode = one_hot_encode
        self.lr = lr
        self.obs_normalization_factor = obs_normalization_factor
        self.clip_value = clip_value

        self.opt = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr)

    def actor_critic_config(
        self,
        actor: nn.Module,
        critic: nn.Module,
        games_to_play: int = 1000,
        one_hot_encode: bool = True,
        gamma: float = 0.99,
        lr: float = 1e-3,
        actor_lr: float = 0,
        critic_lr: float = 0,
        obs_normalization_factor: float = 1,
        clip_value: float = 1,
    ) -> None:

        self.actor = actor
        self.critic = critic
        self.lr = lr
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr if actor_lr > 0 else lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr if critic_lr > 0 else lr)

        self.gamma = gamma
        self.games_to_play = games_to_play
        self.one_hot_encode = one_hot_encode

        self.obs_normalization_factor = obs_normalization_factor
        self.clip_value = clip_value
