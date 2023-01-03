from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import torch

from rldog.dataclasses.generic import GenericConfig
from rldog.tools.logger import logger
from rldog.tools.plotters import plot_results


class BaseAgent(ABC, GenericConfig):
    def __init__(self) -> None:
        self.action_counts = {i: 0 for i in range(self.n_actions)}
        self.evaluation_action_counts = {i: 0 for i in range(self.n_actions)}

        self.reward_averages: list[float] = []
        self.evaluation_reward_averages: list[float] = []
        self.training_loss: List[float] = []
        self.games_played = 0

    def play_games(self, games_to_play: int = 0, verbose: bool = False) -> None:
        """
        Play the games, updating at each step the network if not self.evaluation_mode
        Verbose mode shows some stats at the end of the training, and a graph.
        """
        games_to_play = self.games_to_play if games_to_play == 0 else games_to_play
        game_frac = games_to_play // 10 if games_to_play >= 10 else games_to_play + 1
        mean = lambda lst: sum(lst) / len(lst)
        for game_number in range(games_to_play):
            self._play_game()
            self.games_played += 1  # Needed for epsilon updating
            if game_number % game_frac == 0 and game_number > 0:
                if hasattr(self, "epsilon"):
                    logger.info(
                        f"Played {game_number} games. Epsilon = {self.epsilon:.1f}. Average of last {game_frac} games: reward = {mean(self.reward_averages[-game_frac: ]):.3f}, loss = {mean(self.training_loss[-game_frac: ]):.3f}"
                    )
                else:
                    logger.info(
                        f"Played {game_number} games. Average of last {game_frac} games: reward = {mean(self.reward_averages[-game_frac: ]):.3f}, loss = {mean(self.training_loss[-game_frac: ]):.3f}"
                    )
            while self._network_needs_updating():
                self._update_network()
        if verbose:
            total_rewards = self.reward_averages
            plot_results(total_rewards, title="Training Graph", loss=self.training_loss)

    def evaluate_games(self, games_to_evaluate: int, plot: bool = True) -> None:
        """Evaluate games"""

        for _ in range(games_to_evaluate):
            self._evaluate_game()

        total_rewards = self.evaluation_reward_averages
        if plot:
            plot_results(total_rewards, title="Evaluation")
        logger.info(
            f"Evaluation action counts = {self.evaluation_action_counts}",
        )
        logger.info(f"Mean evaluation reward =  {sum(total_rewards) / len(total_rewards)}")

    def _evaluate_game(self) -> None:
        """
        Evaluates the models performance for one game. Seperate function as this
        runs quicker, at the price of not storing transitions.

        Runs when self.evaluate_games() is called
        """
        next_obs_unformatted, info = self.env.reset()
        next_obs, legal_moves = self._format_obs(next_obs_unformatted, info)
        terminated = False
        rewards = []
        actions = []
        while not terminated:
            obs = next_obs
            action = self._get_action(obs, legal_moves, evaluate=True)
            next_obs_unformatted, reward, terminated, truncated, info = self.env.step(action)
            next_obs, legal_moves = self._format_obs(next_obs_unformatted, info)
            rewards.append(reward)
            actions.append(action)

            terminated = terminated or truncated

        self.evaluation_reward_averages.append(sum(rewards))
        self._update_action_counts(actions, evaluate=True)

    def _update_action_counts(self, actions: List[int], evaluate: bool = False) -> None:

        if evaluate:
            for action in actions:
                self.evaluation_action_counts[action] = self.evaluation_action_counts.get(action, 0) + 1
        else:
            for action in actions:
                self.action_counts[action] = self.action_counts.get(action, 0) + 1

    def _format_obs(
        self, obs: Union[float, int, Tuple, List], info: Dict[Any, Any]
    ) -> Tuple[torch.Tensor, List[int] | range]:
        """Allow obs to be passed into pytorch model"""
        if info.get("legal_moves", False):
            obs, legal_moves = obs  # type: ignore[misc]
        else:
            legal_moves = range(self.n_actions)

        if self.one_hot_encode:
            if not (isinstance(obs, tuple) or isinstance(obs, list)):
                obs = [obs]
            temp = [0] * self.n_obs
            for i in obs:
                temp[int(i)] = 1
            return torch.tensor(temp, dtype=torch.float32), legal_moves
        else:
            new_obs = torch.tensor(obs, dtype=torch.float32)
            if new_obs.ndimension() < 1:
                new_obs = new_obs.unsqueeze(dim=-1)
            new_obs = new_obs / self.obs_normalization_factor
            return new_obs, legal_moves

    def save_model(self, directory: str) -> None:
        torch.save(self.policy_network.state_dict(), directory)

    @abstractmethod
    def _get_action(self, state: torch.Tensor, legal_moves: List[int] | range, evaluate: bool = False) -> int:
        ...

    @abstractmethod
    def _play_game(self) -> None:
        ...

    @abstractmethod
    def _network_needs_updating(self) -> bool:
        ...

    @abstractmethod
    def _update_network(self) -> None:
        ...
