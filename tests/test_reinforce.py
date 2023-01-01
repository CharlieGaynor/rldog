import random
from typing import Any, List, Tuple

import pytest
import torch
import torch.nn as nn

from rldog.agents.policy_gradients.reinforce import Reinforce
from rldog.configs.FrozenLake_config import FrozenLakeConfig
from rldog.dataclasses.policy_dataclasses import Transition
from rldog.networks.networks import BasicSoftMaxNN

n_obs = 16
n_actions = 4


@pytest.fixture()
def agent() -> Reinforce:
    conf = FrozenLakeConfig(is_slippery=False)
    net = BasicSoftMaxNN(input_size=16, output_size=4)
    conf.reinforce_config(network=net)
    a = Reinforce(conf)  # type: ignore[arg-type]
    a.device = torch.device("cpu")
    return a


@pytest.fixture()
def state() -> torch.Tensor:
    encoded_state = torch.eye(n_obs)[int(3)]
    return encoded_state


@pytest.fixture()
def probs_and_actions() -> Tuple[torch.Tensor, ...]:
    probs = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    actions = torch.tensor([[0, 1, 2]], dtype=torch.long)
    return probs, actions


@pytest.fixture()
def transitions() -> List[Transition]:
    return [
        Transition(
            torch.tensor(random.random()),
            random.random(),
        )
        for _ in range(50)
    ]


@pytest.fixture()
def rewards() -> List[float]:
    return [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.fixture()
def action_probs() -> torch.Tensor:
    return torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5], dtype=torch.float32)


def test__compute_discounted_rewards(agent: Reinforce, rewards: List[float]) -> None:
    agent.gamma = 0
    discounted_rewards_tensor = agent._compute_discounted_rewards(rewards)
    assert torch.equal(discounted_rewards_tensor, torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32))

    agent.gamma = 1
    discounted_rewards_tensor = agent._compute_discounted_rewards(rewards)
    assert torch.equal(discounted_rewards_tensor, torch.tensor([15, 14, 12, 9, 5], dtype=torch.float32))


def test__compute_loss(agent: Reinforce, action_probs: torch.FloatTensor, rewards: List[float]) -> None:
    agent.gamma = 0
    loss = agent._compute_loss(action_probs, rewards)
    loss_values = torch.tensor([-0.6931 * i for i in range(1, 6)], dtype=torch.float32)
    expected_loss = torch.mean(-1 * loss_values)
    assert abs(expected_loss.item() - loss.item()) < 1e-1


def test__attributes_from_transitions(agent: Reinforce, transitions: List[Transition]) -> None:
    action_probs, rewards = agent._attributes_from_transitions(transitions)
    assert action_probs.ndimension() == 1
    assert isinstance(rewards, list)


def test_play_games(agent: Reinforce) -> None:
    agent.play_games(1)
    assert agent.games_played > 0
    assert not agent._network_needs_updating()
    agent.play_games(2)
    assert not agent._network_needs_updating()


def test__get_action(agent: Reinforce, state: torch.FloatTensor) -> None:
    agent.policy_network.l1.bias.data.fill_(0)  # type: ignore
    agent.policy_network.l1.weight.data.fill_(0)  # type: ignore
    actions = [agent._get_action(state, range(agent.n_actions), evaluate=False)[0] for _ in range(100)]  # type: ignore[index]
    assert all([i in actions for i in range(n_actions)])
