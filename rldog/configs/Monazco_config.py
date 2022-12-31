import gym
import torch
import torch.nn as nn

from rldog.configs.base_config import BaseConfig
from rldog.networks.basic_nn import BasicNN


class MonazcoConfig(BaseConfig):
    def __init__(self, is_slippery: bool = False):
        self.n_actions = 16
        self.n_obs = 16
        self.state_type = "DISCRETE"
        self.env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery, disable_env_checker=True)
        self.policy_network: nn.Module = BasicNN(input_size=self.n_obs, output_size=self.n_actions)
