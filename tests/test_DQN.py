import random
from typing import Any, List, Tuple

import pytest
import torch
import torch.nn as nn

from rldog.agents.DQN_based.DQN import DQN
from rldog.configs.FrozenLake_config import FrozenLakeConfig
from rldog.dataclasses.DQN_dataclasses import DQNConfig
from rldog.dataclasses.generic import Transition
from rldog.networks.networks import StandardNN

n_obs = 16
n_actions = 4


@pytest.fixture()
def agent() -> DQN:
    conf = FrozenLakeConfig(is_slippery=False)
    conf.DQN_config()
    a = DQN(conf)  # type: ignore[arg-type]
    a.device = torch.device("cpu")
    return a


@pytest.fixture()
def float_net() -> nn.Module:
    net = StandardNN(input_size=1, output_size=4, hidden_layers=0, hidden_size=16)
    return net


@pytest.fixture()
def state() -> torch.Tensor:
    encoded_state = torch.eye(n_obs)[int(3)]
    return encoded_state


class deterministic_DQN(DQN):

    """Same as DQN but with deterministic actions for easier testing"""

    def __init__(self, config: DQNConfig) -> None:
        super().__init__(config)
        self.fixed_actions = [1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 1, 2, 2]
        self.action_idx = 0

    def _get_action(self, state: torch.Tensor, legal_moves: Any, evaluate: Any = False) -> int:
        action = self.fixed_actions[self.action_idx]
        self.action_idx = (self.action_idx + 1) % len(self.fixed_actions)
        return action


@pytest.fixture()
def deterministic_agent() -> deterministic_DQN:
    conf = FrozenLakeConfig(is_slippery=False)
    conf.DQN_config()
    conf.alpha = 0.5
    conf.gamma = 1
    det_agent = deterministic_DQN(conf)  # type: ignore[arg-type]
    det_agent.device = torch.device("cpu")
    return det_agent


@pytest.fixture()
def target_q_values_input() -> Tuple[torch.Tensor, ...]:
    current_q_values = torch.tensor([[i] for i in range(5)], dtype=torch.float32)
    rewards = torch.tensor([0, -1, 0, 1, 1]).unsqueeze(dim=-1)
    next_obs = torch.tensor([[1 if i in [0, 1] else 0 for i in range(16)] for j in range(5)], dtype=torch.float32)
    terminated = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long).unsqueeze(dim=-1)
    actions = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long).unsqueeze(dim=-1)
    return current_q_values, rewards, next_obs, terminated, actions


@pytest.fixture()
def transitions_float() -> List[Transition]:
    transitions = []
    for _ in range(100):
        transitions.append(
            Transition(
                obs=torch.tensor(random.random() * 10, dtype=torch.float32),
                action=random.randint(0, 3),
                reward=random.random(),
                next_obs=torch.tensor(random.random() * 10, dtype=torch.float32),
                terminated=bool(1 if random.random() > 0.9 else 0),
            )
        )
    return transitions


def test__attributes_from_transitions(agent: DQN, transitions_float: List[Transition], float_net: nn.Module) -> None:
    obs, actions, rewards, next_obs, terminated = agent._attributes_from_transitions(transitions_float)
    agent.one_hot_encode = False
    agent.alpha = 0.5
    agent.policy_network = float_net
    current_q_values = agent._calculate_current_q_values(obs, actions)
    target_q_vals = agent._calculate_target_q_values(current_q_values, rewards, next_obs, terminated)

    assert target_q_vals.shape == torch.Size([100, 1])
    assert current_q_values.shape == torch.Size([100, 1])


def test_load_config(agent: DQN) -> None:
    assert True


def test_play_games(agent: DQN) -> None:
    agent.mini_batch_size = 2
    agent.epsilon = 1
    agent.epsilon_decay = 0.75
    agent.epsilon_grace_period = 0
    agent.play_games(1)
    assert agent.games_played > 0
    assert not agent._network_needs_updating()
    agent.play_games(2)
    assert not agent._network_needs_updating()
    assert agent.epsilon != 1


def test__get_action(agent: DQN, state: torch.FloatTensor) -> None:
    agent.policy_network.l1.bias.data[2] = 1  # type: ignore
    action = agent._get_action(state, range(agent.n_actions), evaluate=True)
    assert action == 2
    agent.epsilon = 1
    agent.epsilon_decay = 0
    actions = [agent._get_action(state, range(agent.n_actions), evaluate=False) for _ in range(100)]
    assert all([i in actions for i in range(n_actions)])


def test__format_obs(agent: DQN, state: torch.FloatTensor) -> None:
    assert torch.equal(agent._format_obs(3, {})[0], state)
    assert torch.equal(agent._format_obs(3.0000, {})[0], state)
    agent.one_hot_encode = False
    assert torch.equal(agent._format_obs([3, 4], {})[0], torch.tensor([3, 4], dtype=torch.float32))


def test__update_epsilon(agent: DQN) -> None:
    agent.games_played = 1
    agent.games_to_decay_epsilon_for = 2
    agent.epsilon = 1
    agent.epsilon_decay = 0.5
    agent.epsilon_grace_period = 0
    agent._update_epsilon()
    assert agent.epsilon == 0.5


def test__compute_loss(deterministic_agent: deterministic_DQN) -> None:
    da = deterministic_agent
    # Let the agent calculate the loss for one game
    da.mini_batch_size = 1000
    da.play_games(1)
    da.mini_batch_size = len(da.transitions)
    transitions = da._sample_transitions()
    attributes = da._attributes_from_transitions(transitions)
    # Now we calculate it by hand. First we need to know the Q values.
    # All the weights of the target network are 0, and all the bias' are 1
    da.policy_network.l1.weight.data.fill_(0)  # type: ignore
    da.policy_network.l1.bias.data.fill_(1)  # type: ignore
    # Q(s, a) = 1. So use bellman to get new q values
    # Q(s,a) = (1 - alpha) * Q(s,a) + alpha * (reward + gamma * max_a'{Q(s',a')} )"""
    # 1 = 0.5 * 1 + 0.5 * (0 + 1 * 1) = 1 for all events with no reward
    # 1 = 0.5 + 0.5 * 1 for the last event
    # So loss should be 0
    loss = da._compute_loss(*attributes)
    assert loss == 0

    da.policy_network.l1.bias.data.fill_(0)  # type: ignore
    da.alpha = 0.5
    # Now we have 0 = 0 for all except the last event, where it's 0 = 1
    loss = da._compute_loss(*attributes)
    expected_loss = da.alpha**2 / da.mini_batch_size
    assert abs(loss.detach().item() - expected_loss) < 1e-2


def test__calculate_target_q_values(agent: DQN, target_q_values_input: Tuple[torch.Tensor, ...]) -> None:
    print("------")
    agent.one_hot_encode = False
    agent.alpha = 0.5
    agent.policy_network.l1.reset_parameters()  # type: ignore
    current_q_values, rewards, next_obs, terminated, _ = target_q_values_input
    target_q_vals = agent._calculate_target_q_values(current_q_values, rewards, next_obs, terminated)  # type: ignore

    assert target_q_vals.shape == torch.Size([5, 1])


def test__calculate_target_q_values_no_ohe(agent: DQN, target_q_values_input: Tuple[torch.Tensor, ...]) -> None:
    print("------")
    agent.one_hot_encode = False
    agent.alpha = 0.5
    agent.policy_network.l1.reset_parameters()  # type: ignore
    current_q_values, rewards, next_obs, terminated, _ = target_q_values_input
    target_q_vals = agent._calculate_target_q_values(current_q_values, rewards, next_obs, terminated)  # type: ignore

    assert target_q_vals.shape == torch.Size([5, 1])


def test__calculate_actioned_q_values(agent: DQN, target_q_values_input: Tuple[torch.Tensor, ...]) -> None:
    _, _, current_q_values, _, actions = target_q_values_input
    actioned_q_vals = agent._calculate_actioned_q_values(current_q_values, actions)  # type: ignore
    assert actioned_q_vals.shape == torch.Size([5, 1])
    assert actioned_q_vals.tolist() == [[1.0], [1.0], [0.0], [0.0], [1.0]]


def test__evaluate_game(agent: DQN) -> None:
    agent.evaluate_games(100, plot=False)
