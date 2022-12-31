import random
from collections import deque
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn

from rldog.dataclasses.DQN_dataclasses import DQN_config
from rldog.dataclasses.generic import Transition
from rldog.tools.logger import logger
from rldog.tools.plotters import plot_results


class REINFORCE(nn.Module, DQN_config):
    """
    blah blah blah

    """

    def __init__(self, config: DQN_config, force_cpu: bool = False) -> None:

        super().__init__()
        self.__dict__.update(config.__dict__)

        if force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network.to(self.device)

        self.action_counts = {i: 0 for i in range(self.n_actions)}
        self.evaluation_action_counts = {i: 0 for i in range(self.n_actions)}

        self.transitions: deque[Transition] = deque([], maxlen=self.buffer_size)
        self.reward_averages: list[list[float]] = []
        self.evaluation_reward_averages: list[list[float]] = []
        self.evaluation_mode = False
        self.games_played = 0

    def _play_game(self) -> None:
        """Interact with the environment until 'done' Store transitions in self.transitions & update after every game"""
        next_obs_unformatted = np.array(self.env.reset())
        next_obs = self.format_obs(next_obs_unformatted)
        done = False
        rewards = []
        reward: float = 0

        while not done:
            obs = next_obs
            action = self.get_action(obs)
            next_obs_unformatted, reward, done, termination, _ = self.env.step(action)
            next_obs = self.format_obs(np.array(next_obs_unformatted))
            rewards.append(reward)

            if termination:
                done = True

            self.transitions.append((obs, action, reward))

        if self.evaluation_mode:
            self.evaluation_reward_averages.append([0.0, sum(rewards)])
        else:
            self.reward_averages.append([0.0, sum(rewards)])
            self.update_network()

    def get_action(self, state: torch.Tensor) -> int:
        """Sample actions with softmax probabilities. If evaluating, set a min probability"""

        with torch.no_grad():
            probabilities = self.policy_network(state)

        numpy_probabilities: np.ndarray = probabilities.data.flatten().numpy()

        if self.evaluation_mode:
            trimmed_probs = np.where(numpy_probabilities < 0.05, 0, numpy_probabilities)
            numpy_probabilities = trimmed_probs / np.sum(trimmed_probs)

        action = np.random.choice(len(numpy_probabilities), p=numpy_probabilities)
        return action

    def update_network(self):
        """Sample experiences, compute & back propagate loss"""

        obs = torch.tensor(np.array([s.numpy() for (s, a, r) in self.transitions]), dtype=torch.float32)
        actions = torch.tensor(np.array([a for (s, a, r) in self.transitions]), dtype=torch.long)
        rewards = torch.tensor(np.array([r for (s, a, r) in self.transitions]), dtype=torch.float32)

        loss = self.compute_loss(obs, actions, rewards)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.transitions = []

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.LongTensor,
        rewards: torch.FloatTensor,
    ) -> torch.Tensor:
        """Compute loss according to REINFORCE"""

        probabilities = self.policy_network(obs)
        actioned_probabilities = probabilities.gather(dim=-1, index=actions.view(-1, 1)).squeeze()

        discounted_rewards = self.compute_discounted_rewards(rewards)

        loss = torch.mean(-1.0 * torch.log(actioned_probabilities) * discounted_rewards)
        return loss

    def compute_discounted_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        """Calculate the sum_i^{len(rewards)}r * gamma^i for each time step i"""

        discounted_rewards = []

        for i in range(len(rewards)):
            total = 0
            for j in range(i, len(rewards)):
                total += rewards[j] * self.gamma ** (j)
            discounted_rewards.append(total)

        discounted_rewards_tensor = torch.FloatTensor(discounted_rewards)
        discounted_rewards_tensor /= discounted_rewards_tensor.std() + 1e-3  # type: ignore

        return discounted_rewards_tensor

    def play_games(self, games_to_play: int = 0, verbose: bool = False) -> None:
        """Play the games, updating at each step the network if not self.evaluation_mode
        Verbose mode shows some stats at the end of the training, and a graph."""

        games_to_play = self.max_games if games_to_play == 0 else games_to_play

        while games_to_play > 1:
            self._play_game()
            self.games_played += 1
            games_to_play -= 1

        if verbose:
            if self.evaluation_mode:
                actions = Counter([a for (s, a, r) in self.transitions])
                self.update_action_counts(actions)
                total_rewards = [i[-1] for i in self.evaluation_reward_averages]
                plot_results(total_rewards)
                print("Action counts", self.evaluation_action_counts)
                print("Mean reward", sum(total_rewards) / len(total_rewards))
            else:
                total_rewards = [i[-1] for i in self.reward_averages]
                plot_results(total_rewards)

    def format_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Allow obs to be passed into pytorch model"""
        if self.state_is_discrete:
            encoded_state = np.zeros(self.n_obs, dtype=np.float32)
            encoded_state[obs.item()] = 1
            return torch.tensor(encoded_state, dtype=torch.float32)
        else:
            if len(self.one_hot_encoding_basepoints) > 0:
                indexes = [int(obs[i]) + self.one_hot_encoding_basepoints[i] for i in range(len(obs))]
                encoding = [1 if i in indexes else 0 for i in range(self.n_obs)]
                return torch.tensor(encoding, dtype=torch.float32)

            return torch.tensor(obs, dtype=torch.float32)

    def update_action_counts(self, new_action_counts: Dict[int, int]) -> None:

        if self.evaluation_mode:
            for key, val in new_action_counts.items():
                self.evaluation_action_counts[key] += val
        else:
            for key, val in new_action_counts.items():
                self.action_counts[key] += val

    @staticmethod
    def calculate_actioned_probabilities(probabilities: torch.FloatTensor, actions: torch.LongTensor) -> torch.Tensor:
        """Give me probabilities for all actions, and the actions you took.
        I will return you only the probabilities for the actions you took
        """
        return probabilities[range(probabilities.shape[0]), actions.flatten()]
