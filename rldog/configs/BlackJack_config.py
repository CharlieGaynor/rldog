# import gym
# import torch
# import torch.nn as nn

# from rldog.configs.base_config import BaseConfig
# from rldog.networks.networks import BasicNN


# class FrozenLakeConfig(BaseConfig):
#     def __init__(self, is_slippery: bool = False):
#         self.n_actions = 4
#         self.n_obs = 16
#         self.state_type = "DISCRETE"
#         self.env = gym.make("CliffWalking-v0")
#         self.policy_network: nn.Module = BasicNN(input_size=self.n_obs, output_size=self.n_actions)

#     def DQN_config(self, network: nn.Module = None, max_games: int = 1000) -> None:
#         if network is not None:
#             self.policy_network = network

#         self.lr = 1e-3
#         self.opt = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr)
#         self.max_games: int = max_games
#         self.epsilon = 1.0
#         self.min_epsilon: float = 0.2
#         self.games_to_decay_epsilon_for: int = self.max_games * 3 // 4
#         self.alpha: float = 0.1
#         self.gamma: float = 0.99
#         self.mini_batch_size: int = 4
#         self.buffer_size: int = 128
