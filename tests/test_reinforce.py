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
    net = BasicSoftMaxNN(input_size = 16, output_size = 4)
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
    actions = torch.tensor([[0 , 1, 2]], dtype=torch.long)
    return probs, actions

@pytest.fixture()
def transitions() -> List[Transition]:
    return [Transition(
        torch.tensor([random.random() for _ in range(n_obs)], dtype=torch.float32),
        random.randint(0, 3),
        random.random(),
    ) for _ in range(50)]
    
@pytest.fixture()
def rewards() -> torch.FloatTensor:
    return torch.tensor([[i] for i in range(10)], dtype=torch.float32)

def test__calculate_actioned_probabilities(agent: Reinforce, probs_and_actions: Tuple[torch.Tensor, ...]) -> None:
    expected = torch.tensor([[1, 5, 9]], dtype=torch.float32).squeeze()
    assert torch.equal(agent._calculate_actioned_probabilities(*probs_and_actions), expected)
    
def test__compute_discounted_rewards(agent: Reinforce, rewards: torch.FloatTensor) -> None:
    agent.gamma = 0
    discounted_rewards_tensor = agent._compute_discounted_rewards(rewards)
    
def test__compute_loss() -> ...:
    ...

def test__attributes_from_transitions(agent: Reinforce, transitions: List[Transition]) -> None:
    obs, actions, rewards = agent._attributes_from_transitions(transitions)
    assert obs.ndimension() == 2
    assert actions.ndimension() == 2
    assert rewards.ndimension() == 2

def test_play_games(agent: Reinforce) -> None:
    agent.play_games(1)
    assert agent.games_played > 0
    assert not agent._network_needs_updating()
    agent.play_games(2)
    assert not agent._network_needs_updating()

def test__get_action(agent: Reinforce, state: torch.FloatTensor) -> None:
    agent.policy_network.l1.bias.data.fill_(0)  # type: ignore
    agent.policy_network.l1.weight.data.fill_(0)  # type: ignore
    actions = [agent._get_action(state, range(agent.n_actions), evaluate=False) for _ in range(100)]
    assert all([i in actions for i in range(n_actions)])


# def test__compute_loss(deterministic_agent: deterministic_Reinforce) -> None:
#     da = deterministic_agent
#     da.evaluation_mode = False
#     # Let the agent calculate the loss for one game
#     da.mini_batch_size = 1000
#     da.play_games(1)
#     da.mini_batch_size = len(da.transitions)
#     transitions = da._sample_transitions()
#     attributes = da._attributes_from_transitions(transitions)
#     # Now we calculate it by hand. First we need to know the Q values.
#     # All the weights of the target network are 0, and all the bias' are 1
#     da.policy_network.l1.weight.data.fill_(0)  # type: ignore
#     da.policy_network.l1.bias.data.fill_(1)  # type: ignore
#     # Q(s, a) = 1. So use bellman to get new q values
#     # Q(s,a) = (1 - alpha) * Q(s,a) + alpha * (reward + gamma * max_a'{Q(s',a')} )"""
#     # 1 = 0.5 * 1 + 0.5 * (0 + 1 * 1) = 1 for all events with no reward
#     # 1 = 0.5 + 0.5 * 1 for the last event
#     # So loss should be 0
#     loss = da._compute_loss(*attributes)
#     assert loss == 0

#     da.policy_network.l1.bias.data.fill_(0)  # type: ignore
#     da.alpha = 0.5
#     # Now we have 0 = 0 for all except the last event, where it's 0 = 1
#     loss = da._compute_loss(*attributes)
#     expected_loss = da.alpha**2 / da.mini_batch_size
#     assert abs(loss.detach().item() - expected_loss) < 1e-2
