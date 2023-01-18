import copy
import math
import sys
import time
from collections import Counter
from typing import List

import torch
from torch import nn

from rldog.configs.base_config import BaseConfig
from rldog.dataclasses.policy_dataclasses import PPOConfig, Transition
from rldog.tools.logger import logger
from rldog.tools.plotters import plot_results
from rldog.tools.ppo_experience_generator import ParallelExperienceGenerator


class PPO(PPOConfig):
    """
    Proximal Policy Optimisation. This is has capabality to use multiple workers in parallel.
    The actor & critic do share networks.

    :params:
        config (PPOConfig): Necessary configuration for the agent to run
        force_cpu (bool = False): Use cpu for torch, whether there is access to a gpu or not
    """

    def __init__(self, config: PPOConfig, force_cpu: bool = False) -> None:

        self.__dict__.update(config.__dict__)

        self.device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

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
            self.use_parallel,
            self.n_envs,
        )

        self.transitions: List[Transition] = []
        self.losses: List[float] = []
        self.rewards: List[float] = []

    def play_games(
        self,
        games_to_play: int = 0,
        plot: bool = True,
        log_things: bool = True,
        save_model_at_end: bool = False,
        save_model_at_checkpoints: bool = False,
        model_directory: str = "",
    ) -> None:
        """
        Play {games_to_play} games.

        ::params::
        log_things (bool): Give periodic logs on the progress
        plot (bool): Plot the training results and loss at the end
        save_model_at_checkpoints (bool): Save the model at the end or at checkpoints
        save_model_at_end (bool): ^^
        model_directory (str): Where should we save the model?
        """
        if model_directory == "" and (save_model_at_end or save_model_at_checkpoints):
            logger.warning("Can't save the model as you didn't specifiy a model directory :(")
            save_model_at_checkpoints = False
            save_model_at_end = False

        games_to_play = self.games_to_play if games_to_play == 0 else games_to_play
        number_of__steps = math.ceil(games_to_play / self.n_games_per_learning_batch)
        logger.info(
            f"Playing {number_of__steps * self.n_games_per_learning_batch} games and {number_of__steps} episodes, due to the n_games_per_learning_batch setting"
        )
        games_without_updating = 0

        start_time = time.time()
        for i in range(number_of__steps):
            self._step()
            if log_things:
                last_x_rewards = self.rewards[-self.n_games_per_learning_batch :]
                last_x_losses = self.losses[-self.n_games_per_learning_batch :]
                mean_last_x_rewards = sum(last_x_rewards) / len(last_x_rewards)
                mean_last_x_losses = sum(last_x_losses) / len(last_x_losses)
                logger.info(
                    f"Last {(i + 1) * self.n_games_per_learning_batch} games; Average reward = {mean_last_x_rewards:.2f}, Average loss = {mean_last_x_losses:.2f}"
                )

            if save_model_at_checkpoints:
                games_without_updating += self.n_games_per_learning_batch
                if games_without_updating >= games_to_play / 5:
                    games_without_updating = 0
                    self._save_model(model_directory)

        logger.info(f"Training time: {time.time() - start_time:.2f}s")

        if save_model_at_end or (save_model_at_checkpoints and games_without_updating > 0):
            self._save_model(model_directory)

        if plot:
            plot_results(self.rewards, loss=self.losses, title="PPO training rewards & loss")

    def evaluate_games(
        self, games_to_evaluate: int, plot: bool = True, log_things: bool = True
    ) -> None:
        """
        Evaluate the agent for {games_to_evaluate} games.

        ::params::
        games_to_evaluate (int): How many games to evaluate the agents performance for?
        log_things (bool): Give periodic logs on the progress
        plot (bool): Plot the training results and loss at the end
        """
        evaluation_generator = ParallelExperienceGenerator(
            self.n_actions,
            self.n_obs,
            games_to_evaluate,
            self.one_hot_encode,
            self.obs_normalization_factor,
            self.net,
            self.env,
            device=torch.device("cpu"),
            use_parallel=self.use_parallel,
        )

        if not (log_things or plot):
            logger.warning("There's nothing to show! Set plot or log_things to true")
        else:
            all_states, all_actions, all_rewards = evaluation_generator.play_n_episodes()
            evaluation_rewards = [sum(game_rewards) for game_rewards in all_rewards]
            if log_things:
                actions = [action for epsiode in all_actions for action in epsiode]
                action_counts = {i: 0 for i in range(self.n_actions)}
                action_counts.update(dict(Counter(actions)))
                logger.info(f"Evaluation action_counts = {action_counts}")
                logger.info(
                    f"Average total reward: {sum(evaluation_rewards) / len(evaluation_rewards)}"
                )
            if plot:
                plot_results(test_rewards=evaluation_rewards)

    def _save_model(self, model_directory: str) -> None:
        logger.info(f"Saving model parameters to {model_directory}")
        torch.save(self.net.state_dict(), model_directory)

    def _step(self) -> None:
        """Generate a bunch of data using multiple workers, and learn from it"""
        all_states, all_actions, all_rewards = self.experience_generator.play_n_episodes()
        self.rewards.extend([sum(game_rewards) for game_rewards in all_rewards])
        # Now convert all_states & all actions from List[List[something]] to List[torch.Tensor]
        tensor_states: torch.Tensor = torch.cat(
            [torch.stack(states, dim=0) for states in all_states], dim=0
        )  # Dims: [total_number_of_states, state_size]
        self._policy_learn(tensor_states, all_actions, all_rewards)
        self._equalise_policies()

    def _policy_learn(
        self,
        tensor_states: torch.Tensor,
        all_actions: List[List[int]],
        all_rewards: List[List[float]],
    ) -> None:
        """Learn from the data we've collected, {self.n_learning_episodes_per_batch} many times"""
        all_discounted_returns = self._calculate_all_discounted_returns(all_rewards)

        with torch.no_grad():  # Maybe need this, maybe not
            old_probs: torch.Tensor = self.old_net.forward_policy(tensor_states)
            old_actioned_probs = old_probs[
                range(old_probs.shape[0]), [action for episode in all_actions for action in episode]
            ]

        for _ in range(self.n_learning_episodes_per_batch):
            probs, critic_values = self.net.forward(tensor_states)
            ratio_of_probs = self._calculate_ratio_of_policy_probabilities(
                probs, all_actions, old_actioned_probs
            )
            loss = self._compute_losses(
                ratio_of_probs, all_discounted_returns, torch.squeeze(critic_values), probs
            )
            self._take_policy_new_optimisation_step(loss)

    def _calculate_ratio_of_policy_probabilities(
        self, probs: torch.Tensor, all_actions: List[List[int]], old_actioned_probs: torch.Tensor
    ) -> torch.Tensor:

        actioned_probs = probs[
            range(probs.shape[0]), [action for episode in all_actions for action in episode]
        ]
        ratio_of_policy_probabilities = actioned_probs / (old_actioned_probs + 1e-8)
        return ratio_of_policy_probabilities

    def _compute_losses(
        self,
        ratio_of_probabilities: torch.Tensor,
        discounted_rewards: torch.Tensor,
        critic_values: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PPO loss"""

        ratio_of_probabilities = torch.clamp(
            input=ratio_of_probabilities, min=-sys.maxsize, max=sys.maxsize
        )
        # Can Change the below to introduce an advantage function
        advantage = discounted_rewards - critic_values  # .detach()
        # Clipped surrogate loss
        policy_loss_1 = advantage * ratio_of_probabilities
        policy_loss_2 = advantage * torch.clamp(ratio_of_probabilities, min=1 - 0.2, max=1 + 0.2)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        critic_loss = nn.functional.mse_loss(discounted_rewards, critic_values)
        entropy = (probs * torch.log(probs)).mean()

        loss = policy_loss + critic_loss * 0.5 + entropy * 0.01
        return loss

    def _take_policy_new_optimisation_step(self, loss: torch.Tensor) -> None:
        """Takes an optimisation _step for the new policy"""
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_value)  # type: ignore[attr-defined]
        self.opt.step()

        self.losses.append(loss.item())

    def _calculate_all_discounted_returns(self, all_rewards: List[List[float]]) -> torch.Tensor:
        """Calculate the sum_i^{len(rewards)}r * gamma^i for each time _step i"""

        all_discounted_rewards = []
        for rewards in all_rewards:
            discounted_rewards = [0.0] * len(rewards)

            discounted_rewards[-1] = rewards[-1]
            for idx in reversed(range(len(rewards) - 1)):
                discounted_rewards[idx] = self.gamma * discounted_rewards[idx + 1] + rewards[idx]

            discounted_rewards_tensor = torch.Tensor(discounted_rewards)
            all_discounted_rewards.append(discounted_rewards_tensor)

        return torch.cat(all_discounted_rewards, dim=0)

    def _equalise_policies(self) -> None:
        """Sets the old policy's parameters equal to the new policy's parameters"""
        for old_param, new_param in zip(self.old_net.parameters(), self.net.parameters()):
            old_param.data.copy_(new_param.data)
