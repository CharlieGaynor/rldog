from collections import deque
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn

from rldog.dataclasses.policy_dataclasses import DQN_config, Transition
from rldog.agents.base_agent import BaseAgent
from rldog.tools.logger import logger
from rldog.tools.plotters import plot_results
import random


class Reinforce(BaseAgent, DQN_config):
    """
    blah blah blah

    """

    def __init__(self, config: DQN_config, force_cpu: bool = False) -> None:

        self.__dict__.update(config.__dict__)
        super().__init__()
        

        if force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network.to(self.device)
        self.transitions: List[Transition] = []

    def _play_game(self) -> None:
        """
        Interact with the environment until 'terminated'
        store transitions in self.transitions & updates
        epsilon after each game has finished
        """
        next_obs_unformatted, info = self.env.reset()
        next_obs, legal_moves = self._format_obs(next_obs_unformatted, info)
        terminated = False
        rewards = []
        while not terminated:
            obs = next_obs
            action = self._get_action(obs, legal_moves)
            next_obs_unformatted, reward, terminated, truncated, info = self.env.step(action)
            next_obs, legal_moves = self._format_obs(next_obs_unformatted, info)
            rewards.append(reward)
            self.transitions.append(Transition(obs, action, reward))
            terminated = terminated or truncated

        self.reward_averages.append(sum(rewards))

    def _get_action(self, state: torch.Tensor, legal_moves: List[int] | range, evaluate: bool = False) -> int:
        """Sample actions with softmax probabilities. If evaluating, set a min probability"""

        with torch.no_grad():
            probabilities = self.policy_network(state.to(self.device))

        probs = probabilities.tolist()
        if len(legal_moves) < len(probabilities):
        
            legal_probs = [probs[i] for i in legal_moves]

            action = random.choices(legal_moves, weights=legal_probs, k=1)[0]
        else:

            action = random.choices(range(len(probs)), weights=probs, k=1)[0]
        return action

    def _update_network(self):
        """Sample experiences, compute & back propagate loss"""

        attributes = self._attributes_from_transitions(self.transitions)

        loss = self._compute_loss(*attributes)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        self.transitions = []

    def _network_needs_updating(self):
        return len(self.transitions) > 0

    def _compute_loss(
        self,
        obs: torch.FloatTensor,
        actions: torch.LongTensor,
        rewards: List[float],
    ) -> torch.Tensor:
        """Compute loss according to REINFORCE"""

        probabilities = self.policy_network(obs.to(self.device))
        actioned_probabilities = probabilities.gather(dim=-1, index=actions.view(-1, 1)).squeeze()

        discounted_rewards = self._compute_discounted_rewards(rewards)

        loss = torch.mean(-1.0 * torch.log(actioned_probabilities) * discounted_rewards)
        return loss

    def _compute_discounted_rewards(self, rewards: List[float]) -> torch.FloatTensor:
        """Calculate the sum_i^{len(rewards)}r * gamma^i for each time step i"""

        discounted_rewards = [0] * len(rewards)
        
        discounted_rewards[-1] = rewards[-1]
        for idx in reversed(range(len(rewards) - 1)):
            discounted_rewards[idx] = discounted_rewards[idx + 1] + self.gamma * rewards[idx]

        mean_val = sum(discounted_rewards) / len(discounted_rewards)
        discounted_rewards_tensor = torch.FloatTensor( [i - mean_val for i in discounted_rewards])

        return discounted_rewards_tensor

    @staticmethod
    def _calculate_actioned_probabilities(probabilities: torch.FloatTensor, actions: torch.LongTensor) -> torch.Tensor:
        """Give me probabilities for all actions, and the actions you took.
        I will return you only the probabilities for the actions you took
        """
        return probabilities[range(probabilities.shape[0]), actions.flatten()]

    @staticmethod
    def _attributes_from_transitions(
        transitions: List[Transition],
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, List[float]]:
        """
        Extracts, transforms (and loads, hehe) the attributes hidden in within transitions
        Each resulting tensor should have shape [batch_size, attribute size]
        """

        obs_list = [transition.obs for transition in transitions]
        actions_list = [transition.action for transition in transitions]
        rewards_list = [transition.reward for transition in transitions]

        obs: torch.FloatTensor = torch.stack(obs_list, dim=0)  # type: ignore[assignment]
        # Below might need changing when we consider non integer actions?
        actions: torch.LongTensor = torch.tensor(actions_list, dtype=torch.long).unsqueeze(dim=-1)  # type: ignore[assignment]
        # rewards: List[float] = torch.tensor(rewards_list).unsqueeze(dim=-1)  # type: ignore[assignment]

        while obs.ndimension() < 2:
            obs = obs.unsqueeze(dim=-1)  # type: ignore[assignment]

        return obs, actions, rewards_list
