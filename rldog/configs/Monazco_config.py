import gym

from rldog.configs.base_config import BaseConfig


class MonazcoConfig(BaseConfig):
    def __init__(self, is_slippery: bool = False):
        self.n_actions = 16
        self.n_obs = 16
        self.state_type = "DISCRETE"
        self.env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery, disable_env_checker=True)
