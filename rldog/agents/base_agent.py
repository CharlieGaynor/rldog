from abc import ABC
from rldog.tools.logger import logger
from rldog.tools.plotters import plot_results
import torch
from typing import List, Tuple
from rldog.dataclasses.generic import Transition

class BaseAgent(ABC):
    
    def __init__(self) -> None:
        ...

    def play_games(self, games_to_play: int = 0, verbose: bool = False) -> None:
        """
        Play the games, updating at each step the network if not self.evaluation_mode
        Verbose mode shows some stats at the end of the training, and a graph.
        """
        games_to_play = self.games_to_play if games_to_play == 0 else games_to_play
        game_frac = games_to_play // 10 if games_to_play >= 10 else self.games_to_play + 1
        mean = lambda lst: sum(lst) / len(lst)
        for game_number in range(games_to_play):
            self._play_game()
            self.games_played += 1  # Needed for epsilon updating
            if game_number % game_frac == 0 and game_number > 0:
                logger.info(
                    f"Played {game_number} games. Epsilon = {self.epsilon}. Average reward of last {game_frac} games = {mean(self.reward_averages[-game_frac: ])}"
                )
            while self._network_needs_updating():
                self._update_network()
        if verbose:
            total_rewards = self.reward_averages
            plot_results(total_rewards, title="Training Graph")

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

    def save_model(self, directory: str) -> None:
        torch.save(self.policy_network.state_dict(), directory)