from collections import deque
from typing import Any, Dict, List, Tuple, Union, overload
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
            action, action_probs = self._get_action(obs, legal_moves)  # type: ignore[misc]
            next_obs_unformatted, reward, terminated, truncated, info = self.env.step(action)
            next_obs, legal_moves = self._format_obs(next_obs_unformatted, info)
            rewards.append(reward)
            self.transitions.append(Transition(action_probs, reward))
            terminated = terminated or truncated

        self.reward_averages.append(sum(rewards))

    def _get_action(
        self, state: torch.Tensor, legal_moves: List[int] | range, evaluate: bool = False
    ) -> Union[int, Tuple[int, torch.Tensor]]:  # type:ignore[override]
        """Sample actions with softmax probabilities. If evaluating, set a min probability"""

        if evaluate:
            with torch.no_grad():
                probabilities: torch.FloatTensor = self.policy_network(state.to(self.device))
        else:
            probabilities: torch.FloatTensor = self.policy_network(state.to(self.device))  # type: ignore[no-redef]

        probs = probabilities.tolist()
        if len(legal_moves) < len(probabilities):

            legal_probs = [probs[i] for i in legal_moves]
            action = random.choices(legal_moves, weights=legal_probs, k=1)[0]
        else:
            action = random.choices(range(len(probs)), weights=probs, k=1)[0]

        if evaluate:
            return action
        else:
            return action, probabilities[action]

    def _update_network(self) -> None:
        """Sample experiences, compute & back propagate loss"""

        attributes = self._attributes_from_transitions(self.transitions)

        loss = self._compute_loss(*attributes)
        self.opt.zero_grad()
        loss.backward()

        nn.utils.clip_grad_value_(self.policy_network.parameters(), self.clip_value)  # type: ignore[attr-defined]
        self.opt.step()

        self.transitions = []
        self.training_loss.append(loss.item())

    def _network_needs_updating(self) -> bool:
        return len(self.transitions) > 0

    def _compute_loss(
        self,
        action_probs: torch.FloatTensor,
        rewards: List[float],
    ) -> torch.Tensor:
        """Compute loss according to REINFORCE

        Intuitive explanation for loss function:

        We are minimising -1.0 * torch.log(action_probs) <=> maximise torch.log(action_probs)
        <=> make action_probs -> 1. The reward acts as a multiplier. If we get a high reward,
        then maximise the probs with more urgency, and if we get a lower reward, maximise with less urgency.

        Eventually the higher reward actions are closer to 1, and the lower rewards closer to 0 (since they must sum to 1)"""

        discounted_rewards = self._compute_discounted_rewards(rewards)

        loss = torch.mean(-1.0 * torch.log(action_probs) * discounted_rewards)
        return loss

    def _compute_discounted_rewards(self, rewards: List[float]) -> torch.FloatTensor:
        """Calculate the sum_i^{len(rewards)}r * gamma^i for each time step i"""

        discounted_rewards = [0.0] * len(rewards)

        discounted_rewards[-1] = rewards[-1]
        for idx in reversed(range(len(rewards) - 1)):
            discounted_rewards[idx] = self.gamma * discounted_rewards[idx + 1] + rewards[idx]

        discounted_rewards_tensor = torch.FloatTensor(discounted_rewards)  # norm_rewards )

        return discounted_rewards_tensor

    @staticmethod
    def _attributes_from_transitions(
        transitions: List[Transition],
    ) -> Tuple[torch.FloatTensor, List[float]]:
        """
        Extracts, transforms (and loads, hehe) the attributes hidden in within transitions
        Each resulting tensor should have shape [batch_size, attribute size]
        """
        actions_list = [transition.action_probs for transition in transitions]
        rewards_list = [transition.reward for transition in transitions]

        # Stack because actions_list contains a list of 0 dimensional tensors
        # action_probs has dimensions [num_actions]
        action_probs: torch.FloatTensor = torch.stack(actions_list, dim=0)  # type: ignore[assignment]

        return action_probs, rewards_list
