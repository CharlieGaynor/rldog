import pytest
import torch

from rldog.configs.FrozenLake_config import FrozenLakeConfig
from rldog.networks.networks import StandardPPO
from rldog.tools.ppo_experience_generator import ParallelExperienceGenerator


@pytest.fixture
def net() -> StandardPPO:
    test_net = StandardPPO(input_size=16, output_size=4, hidden_size=1, hidden_layers=1)
    return test_net


@pytest.fixture
def generator(net: StandardPPO) -> ParallelExperienceGenerator:

    conf = FrozenLakeConfig(is_slippery=False)

    generator = ParallelExperienceGenerator(
        n_actions=4,
        n_obs=16,
        n_episodes=1,
        one_hot_encode=True,
        obs_norm_factor=1,
        net=net,
        env=conf.env,
        device=torch.device("cpu"),
        use_parallel=False,
        n_envs=1,
    )
    return generator


def test_play_n_episodes(generator: ParallelExperienceGenerator) -> None:
    generator.use_parallel = False  # Just for safety
    n_eps = 1
    generator.n_episodes = n_eps
    all_states, all_actions, all_rewards = generator.play_n_episodes()
    assert len(all_states) == len(all_actions) == len(all_rewards) == n_eps

    generator.use_parallel = True
    generator.n_episodes = n_eps
    generator.n_envs = 4
    all_states2, all_actions2, all_rewards2 = generator.play_n_episodes()
    assert len(all_states2) == len(all_actions2) == len(all_rewards2) == n_eps
