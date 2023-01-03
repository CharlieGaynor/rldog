import logging
from typing import List

import pytest
import torch

from rldog.agents.policy_gradients.PPO import PPO
from rldog.configs.FrozenLake_config import FrozenLakeConfig
from rldog.networks.networks import StandardPPO
from rldog.tools.logger import logger

n_obs = 16
n_actions = 4


@pytest.fixture
def net() -> StandardPPO:
    test_net = StandardPPO(input_size=16, output_size=4, hidden_size=1, hidden_layers=1)
    for layer in test_net.layers:
        layer.bias.data.fill_(0)
        layer.weight.data.fill_(0)
    test_net.value_head.weight.data.fill_(0)
    test_net.policy_head.weight.data.fill_(0)
    test_net.value_head.bias.data.fill_(0)
    test_net.policy_head.bias.data.fill_(0)
    return test_net


@pytest.fixture()
def agent(net: StandardPPO) -> PPO:
    conf = FrozenLakeConfig(is_slippery=False)
    conf.PPO_config(
        net=net,
        old_net=net,
        one_hot_encode=True,
        n_games_per_learning_batch=1,
        n_learning_episodes_per_batch=1,
        gamma=0,
    )
    agent = PPO(conf, force_cpu=True)
    return agent


@pytest.fixture()
def states() -> torch.Tensor:
    one_obs = torch.eye(n_obs)[int(0)]
    all_states = [[one_obs for _ in range(2)] for _ in range(2)]
    tensor_states: torch.Tensor = torch.cat([torch.stack(states, dim=0) for states in all_states], dim=0)
    return tensor_states


@pytest.fixture()
def actions() -> List[List[int]]:
    actions = [[0 for _ in range(2)] for _ in range(2)]
    return actions


@pytest.fixture()
def rewards() -> List[List[float]]:
    rewards = [[1.0 for _ in range(2)] for _ in range(2)]
    return rewards


@pytest.fixture()
def action_probs() -> torch.Tensor:
    return torch.tensor([[0.5, 0.5, 0.5, 0.5] for _ in range(4)], dtype=torch.float32)


@pytest.fixture()
def old_probs() -> torch.Tensor:
    return torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)


@pytest.fixture()
def ratio_of_probs() -> torch.Tensor:
    return torch.tensor([1, 1, 1, 1], dtype=torch.float32)


@pytest.fixture()
def critic_values() -> torch.Tensor:
    return torch.tensor([[0] for _ in range(4)], dtype=torch.float32)


def test_play_games(agent: PPO) -> None:
    logger.setLevel(logging.WARNING)
    agent.n_games_per_learning_batch = 1
    agent.n_learning_episodes_per_batch = 1
    agent.play_games(1, plot=False, log_things=False)
    agent.play_games(2, plot=False, log_things=False)
    logger.setLevel(logging.DEBUG)


def test__calculate_ratio_of_policy_probabilities(
    agent: PPO, action_probs: torch.Tensor, actions: List[List[int]], old_probs: torch.Tensor
) -> None:
    agent.gamma = 0
    ratio_of_probs = agent._calculate_ratio_of_policy_probabilities(action_probs, actions, old_probs)
    assert torch.equal(ratio_of_probs, torch.tensor([1, 1, 1, 1], dtype=torch.float32))

    ratio_of_probs = agent._calculate_ratio_of_policy_probabilities(action_probs, actions, old_probs / 2)
    assert torch.equal(ratio_of_probs, torch.tensor([2, 2, 2, 2], dtype=torch.float32))


def test__calculate_all_discounted_returns(agent: PPO, rewards: List[List[float]]) -> None:
    dr = agent._calculate_all_discounted_returns(rewards)
    assert torch.equal(dr, torch.tensor([1.0 for _ in range(4)], dtype=torch.float32))


def test__compute_loss(
    agent: PPO,
    ratio_of_probs: torch.Tensor,
    rewards: List[List[float]],
    action_probs: torch.Tensor,
    critic_values: torch.Tensor,
) -> None:
    agent.gamma = 0
    discounted_rewards = agent._calculate_all_discounted_returns(rewards)
    loss = agent._compute_losses(ratio_of_probs, discounted_rewards, torch.squeeze(critic_values), action_probs)
    assert abs(loss - (-1 + 1 * 0.5 + 0.01 * -0.3466)) < 1e-1


def test__policy_learn(agent: PPO, states: torch.Tensor, actions: List[List[int]], rewards: List[List[float]]) -> None:
    agent.gamma = 0
    agent._policy_learn(states, actions, rewards)
