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
