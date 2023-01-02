from collections import deque
from typing import Any, Dict, List, Tuple, Union, overload
import torch
from torch import nn
import sys
from rldog.dataclasses.policy_dataclasses import PPOConfig, Transition
from rldog.agents.base_agent import BaseAgent
from rldog.tools.ppo_experience_generator import ParallelExperienceGenerator
from rldog.tools.logger import logger
import random
import copy
import math

from rldog.tools.plotters import plot_results


class PPO(PPOConfig):
    """
    blah blah blah

            # Then try to test and see where it fails. Once it can solve frozen lake with no slippery,
            # write unit tests. After that, try to compare it to other PPO benchmarks
    """

    def __init__(self, config: PPOConfig, force_cpu: bool = False) -> None:

        self.__dict__.update(config.__dict__)

        if force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.old_net.load_state_dict(copy.deepcopy(self.net.state_dict()))

        self.net.to(self.device)
        self.old_net.to(self.device)

        self.experience_generator = ParallelExperienceGenerator(
            self.n_actions,
            self.n_obs,
            self.n_games_per_learning_batch,
            self.one_hot_encode,
            self.obs_normalization_factor,
            self.net,
            self.env,
            self.device,
        )

        self.transitions: List[Transition] = []
        self.losses: List[float] = []
        self.rewards: List[float] = []

    def play_games(self, games_to_play: int = 0, plot: bool = False) -> None:

        games_to_play = self.games_to_play if games_to_play == 0 else games_to_play
        number_of_steps = math.ceil(games_to_play / self.n_games_per_learning_batch)
        logger.info(
            f"Playing {number_of_steps * self.n_games_per_learning_batch} games and {number_of_steps} episodes, due to the n_games_per_learning_batch setting"
        )
        for i in range(number_of_steps):
            self.step()
            last_x_rewards = self.rewards[-self.n_games_per_learning_batch :]
            last_x_losses = self.losses[-self.n_games_per_learning_batch :]
            mean_last_x_rewards = sum(last_x_rewards) / len(last_x_rewards)
            mean_last_x_losses = sum(last_x_losses) / len(last_x_losses)
            logger.info(
                f"Last {(i + 1) * self.n_games_per_learning_batch} games; Average reward = {mean_last_x_rewards:.2f}, Average loss = {mean_last_x_losses:.2f}"
            )

        if plot:
            plot_results(self.rewards, loss=self.losses, title="PPO training rewards & loss")

    def step(self) -> None:
        all_states, all_actions, all_rewards = self.experience_generator.play_n_episodes()
        self.rewards.extend([sum(game_rewards) for game_rewards in all_rewards])
        # Now convert all_states & all actions from List[List[something]] to List[torch.Tensor]
        tensor_states: torch.Tensor = torch.cat(
            [torch.stack(states, dim=0) for states in all_states], dim=0
        )  # Dims: [total_number_of_states, state_size]
        self.policy_learn(tensor_states, all_actions, all_rewards)
        self.equalise_policies()

    def policy_learn(self, tensor_states: torch.Tensor, all_actions: List[List[int]], all_rewards):
        """A learning iteration for the policy"""
        all_discounted_returns = self._calculate_all_discounted_returns(all_rewards)

        with torch.no_grad():  # Maybe need this, maybe not
            old_probs: torch.Tensor = self.old_net.forward_policy(tensor_states)
            old_actioned_probs = old_probs[
                range(old_probs.shape[0]), [action for episode in all_actions for action in episode]
            ]

        for _ in range(self.n_learning_episodes_per_batch):
            probs, critic_values = self.net.forward(tensor_states)
            ratio_of_probs = self._calculate_ratio_of_policy_probabilities(probs, all_actions, old_actioned_probs)
            policy_loss, critic_loss = self._compute_losses(
                ratio_of_probs, all_discounted_returns, torch.squeeze(critic_values)
            )
            self.take_policy_new_optimisation_step(policy_loss, critic_loss)

    def _calculate_ratio_of_policy_probabilities(
        self, probs: torch.Tensor, all_actions: List[List[int]], old_actioned_probs
    ):

        actioned_probs = probs[range(probs.shape[0]), [action for episode in all_actions for action in episode]]
        ratio_of_policy_probabilities = actioned_probs / (old_actioned_probs + 1e-8)
        return ratio_of_policy_probabilities

    def _compute_losses(
        self,
        ratio_of_probabilities: torch.Tensor,
        discounted_rewards: torch.FloatTensor,
        critic_values: torch.FloatTensor,
    ) -> torch.Tensor:
        """Compute PPO loss"""

        ratio_of_probabilities = torch.clamp(input=ratio_of_probabilities, min=-sys.maxsize, max=sys.maxsize)
        # Can Change the below to introduce an advantage function
        advantage = discounted_rewards - critic_values  # .detach()
        # policy_loss_sum = torch.clamp(ratio_of_probabilities, min=0.8, max=1.2) * advantage  # why does this suck?
        policy_loss_sum = torch.where(
            advantage > 0,
            torch.clamp(ratio_of_probabilities, max=1.2) * advantage,
            torch.clamp(ratio_of_probabilities, min=0.8) * advantage,
        )

        policy_loss = -torch.mean(policy_loss_sum)
        critic_loss = nn.functional.mse_loss(discounted_rewards, critic_values)
        return policy_loss, critic_loss

    def take_policy_new_optimisation_step(self, policy_loss: torch.Tensor, critic_loss: torch.Tensor):
        """Takes an optimisation step for the new policy"""
        self.opt.zero_grad()
        loss = policy_loss + critic_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()

        self.losses.append(loss.item())

    def _calculate_all_discounted_returns(self, all_rewards: List[List[float]]) -> List[torch.FloatTensor]:
        """Calculate the sum_i^{len(rewards)}r * gamma^i for each time step i"""

        all_discounted_rewards = []
        for rewards in all_rewards:
            discounted_rewards = [0.0] * len(rewards)

            discounted_rewards[-1] = rewards[-1]
            for idx in reversed(range(len(rewards) - 1)):
                discounted_rewards[idx] = self.gamma * discounted_rewards[idx + 1] + rewards[idx]

            discounted_rewards_tensor = torch.FloatTensor(discounted_rewards)
            all_discounted_rewards.append(discounted_rewards_tensor)

        return torch.cat(all_discounted_rewards, dim=0)

    def equalise_policies(self):
        """Sets the old policy's parameters equal to the new policy's parameters"""
        for old_param, new_param in zip(self.old_net.parameters(), self.net.parameters()):
            old_param.data.copy_(new_param.data)
