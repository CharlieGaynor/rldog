import random
from typing import List

import pytest
import torch

from rldog.agents.actor_critics.A2C import A2C
from rldog.configs.FrozenLake_config import FrozenLakeConfig
from rldog.dataclasses.actor_critic_dataclasses import Transition
from rldog.networks.networks import BasicNN, BasicSoftMaxNN

n_obs = 16
n_actions = 4


@pytest.fixture()
def agent() -> A2C:
    conf = FrozenLakeConfig(is_slippery=False)
    actor = BasicSoftMaxNN(input_size=16, output_size=4)
    critic = BasicNN(input_size=16, output_size=1)
    conf.actor_critic_config(actor=actor, critic=critic)
    a = A2C(conf)  # type: ignore[arg-type]
    a.device = torch.device("cpu")
    return a


@pytest.fixture()
def state() -> torch.Tensor:
    encoded_state = torch.eye(n_obs)[int(3)]
    return encoded_state


@pytest.fixture()
def transitions() -> List[Transition]:
    return [
        Transition(
            torch.tensor(random.random()),
            random.random(),
            torch.tensor(random.random()).unsqueeze(dim=-1),  # type: ignore[arg-type]
        )
        for _ in range(50)
    ]


@pytest.fixture()
def rewards() -> List[float]:
    return [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.fixture()
def action_probs() -> torch.Tensor:
    return torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5], dtype=torch.float32)


@pytest.fixture()
def critic_values() -> torch.Tensor:
    return torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32)


def test__compute_losses(
    agent: A2C, action_probs: torch.FloatTensor, rewards: List[float], critic_values: torch.FloatTensor
) -> None:
    agent.gamma = 0
    actor_loss, critic_loss = agent._compute_losses(action_probs, rewards, critic_values)
    actor_loss_values = torch.tensor([-0.6931 * (i - 1) for i in range(1, 6)], dtype=torch.float32)
    expected_actor_loss = torch.sum(-1 * actor_loss_values)
    assert abs(expected_actor_loss.item() - actor_loss.item()) < 1e-1

    critic_loss_values = torch.tensor([(i - 1) ** 2 for i in range(1, 6)], dtype=torch.float32)
    expected_critic_loss = torch.sum(critic_loss_values)
    assert abs(expected_critic_loss.item() - critic_loss.item()) < 1e-1


def test__attributes_from_transitions(agent: A2C, transitions: List[Transition]) -> None:
    action_probs, rewards, critic_values = agent._attributes_from_transitions(transitions)
    assert action_probs.ndimension() == 1
    assert isinstance(rewards, list)
    assert critic_values.ndimension() == 1  # Should it be 1 or two?


def test_play_games(agent: A2C) -> None:
    agent.play_games(1)
    assert agent.games_played > 0
    assert not agent._network_needs_updating()
    agent.play_games(2)
    assert not agent._network_needs_updating()


def test__get_action(agent: A2C, state: torch.FloatTensor) -> None:
    agent.actor.l1.bias.data.fill_(0)  # type: ignore
    agent.actor.l1.weight.data.fill_(0)  # type: ignore
    actions = [agent._get_action(state, range(agent.n_actions), evaluate=False)[0] for _ in range(100)]  # type: ignore[index]
    assert all([i in actions for i in range(n_actions)])
