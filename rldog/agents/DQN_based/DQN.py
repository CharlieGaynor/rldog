import random
from collections import deque
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn

from rldog.dataclasses.DQN_dataclasses import DQN_config
from rldog.dataclasses.generic import Transition
from rldog.tools.logger import logger
from rldog.tools.plotters import plot_results

# at the minute using epislon greedy - could generalise this out into a seperate class
# priority is having mini batches


class DQN(nn.Module, DQN_config):
    """
    Deep Q-learning agent, capable of learning in a variety of envrionments. Doesn't recieve legal moves for now

    args:
        config (object): Bunch of hyperparameters and necessary metadata for each
        environment.

    """

    def __init__(self, config: DQN_config, force_cpu: bool = False) -> None:

        # Could error here due to which parent to initialise?
        super().__init__()
        self.__dict__.update(config.__dict__)

        if force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network.to(self.device)

        self.epsilon_decay: float = (self.epsilon - self.min_epsilon) / self.games_to_decay_epsilon_for
        self.action_counts = {i: 0 for i in range(self.n_actions)}
        self.evaluation_action_counts = {i: 0 for i in range(self.n_actions)}
        self.state_is_discrete: bool = self.state_type == "DISCRETE"

        self.transitions: deque[Transition] = deque([], maxlen=self.buffer_size)
        self.reward_averages: list[float] = []
        self.evaluation_reward_averages: list[float] = []
        self.evaluation_mode = False
        self.games_played = 0

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

    def save_model(self, directory: str) -> None:
        torch.save(self.policy_network.state_dict(), directory)

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
            self.transitions.appendleft(Transition(obs, action, reward, next_obs, terminated))
            terminated = terminated or truncated

        self._update_epsilon()
        self.reward_averages.append(sum(rewards))

    def _get_action(self, state: torch.Tensor, legal_moves: List[int] | range, evaluate: bool = False) -> int:
        """Sample actions with epsilon-greedy policy
        use epsilon = 0 for evaluation mode"""

        if not evaluate:
            if random.random() < self.epsilon:
                if legal_moves is None:
                    return random.choice(legal_moves)
                return random.choice(legal_moves)

        with torch.no_grad():
            q_values: torch.FloatTensor = self.policy_network(state.to(self.device))
            qvals = q_values.tolist()

        max_val = float("-inf")
        max_idx = 0
        for idx in legal_moves:
            if qvals[idx] > max_val:
                max_val = qvals[idx]
                max_idx = idx
        return max_idx

    def _update_network(self) -> None:
        """
        Sample transitions, compute & back propagate loss,
        and call itself recursively if there are still samples to train on
        """
        transitions = self._sample_transitions()  # shape [mini_batch_size]
        # Attributes are obs, actions, rewards, next_obs, terminated
        attributes = self._attributes_from_transitions(transitions)
        loss = self._compute_loss(*attributes)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self._update_action_counts(attributes[1].flatten().tolist())

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

    def _network_needs_updating(self) -> bool:
        """
        For standard DQN, network needs updated if self.transitions contains more than
        self.mini_batch_size items
        """
        return len(self.transitions) >= self.mini_batch_size

    def _sample_transitions(self) -> List[Transition]:
        """
        Returns list of transitions with dimensions [mini_batch_size]
        """
        return [self.transitions.pop() for _ in range(self.mini_batch_size)]

    def _update_action_counts(self, actions: List[int], evaluate: bool = False) -> None:

        if evaluate:
            for action in actions:
                self.evaluation_action_counts[action] = self.evaluation_action_counts.get(action, 0) + 1
        else:
            for action in actions:
                self.action_counts[action] = self.action_counts.get(action, 0) + 1

    def _compute_loss(
        self,
        obs: torch.FloatTensor,
        actions: torch.LongTensor,
        rewards: torch.FloatTensor,
        next_obs: torch.Tensor,
        terminated: torch.LongTensor,
    ) -> torch.Tensor:
        """Compute loss according to Q learning equation, and return MSE"""

        current_q_vals = self._calculate_current_q_values(obs, actions)
        with torch.no_grad():
            target_q_vals = self._calculate_target_q_values(current_q_vals, rewards, next_obs, terminated)

        loss = torch.mean((target_q_vals - current_q_vals) ** 2)
        return loss

    def _calculate_current_q_values(self, obs: torch.FloatTensor, actions: torch.LongTensor) -> torch.FloatTensor:
        """Computes the Q values for the actions we took"""
        q_values = self.policy_network(obs)
        actioned_q_values = self._calculate_actioned_q_values(q_values, actions)
        return actioned_q_values

    def _calculate_target_q_values(
        self,
        current_q_vals: torch.Tensor,
        rewards: torch.FloatTensor,
        next_obs: torch.Tensor,
        terminated: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Computes the 'Target' Q values for the actions we took
        Q(s,a) = (1 - alpha) * Q(s,a) + alpha * (reward + gamma * max_a'{Q(s',a')} )"""

        next_q_vals: torch.FloatTensor = self.policy_network(next_obs.to(self.device)).detach()
        target_q_vals_max: torch.FloatTensor = torch.max(next_q_vals, dim=-1).values.unsqueeze(dim=-1)

        # What should the Q values be updated to for the actions we took?
        target_q_vals: torch.FloatTensor = (
            current_q_vals * (1 - self.alpha)
            + self.alpha * rewards
            + self.alpha * self.gamma * target_q_vals_max * (1 - terminated)
        )
        return target_q_vals

    @staticmethod
    def _calculate_actioned_q_values(q_vals: torch.FloatTensor, actions: torch.LongTensor) -> torch.FloatTensor:
        """Give me Q values for all actions, and the actions you took.
        I will return you only the Q values for the actions you took
        """
        actioned_q_vals: torch.FloatTensor = q_vals[range(q_vals.shape[0]), actions.flatten()]  # type: ignore[assignment]
        return actioned_q_vals.unsqueeze(dim=-1)  # type: ignore[return-value]

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

    def _update_epsilon(self) -> None:
        """Slowly decrease epsilon to reduce exploration"""
        if (
            self.games_played > self.epsilon_grace_period
            and (self.games_played - self.epsilon_grace_period) < self.games_to_decay_epsilon_for
        ):
            self.epsilon -= self.epsilon_decay

    @staticmethod
    def _attributes_from_transitions(
        transitions: List[Transition],
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        """
        Extracts, transforms (and loads, hehe) the attributes hidden in within transitions
        Each resulting tensor should have shape [batch_size, attribute size]
        """

        obs_list = [transition.obs for transition in transitions]
        actions_list = [transition.action for transition in transitions]
        rewards_list = [transition.reward for transition in transitions]
        next_obs_list = [transition.next_obs for transition in transitions]
        terminated_list = [transition.terminated for transition in transitions]

        obs: torch.FloatTensor = torch.stack(obs_list, dim=0)  # type: ignore[assignment]
        # Below might need changing when we consider non integer actions?
        actions: torch.LongTensor = torch.tensor(actions_list, dtype=torch.long).unsqueeze(dim=-1)  # type: ignore[assignment]
        rewards: torch.FloatTensor = torch.tensor(rewards_list).unsqueeze(dim=-1)  # type: ignore[assignment]
        next_obs: torch.FloatTensor = torch.stack(next_obs_list, dim=0)  # type: ignore[assignment]
        terminated: torch.LongTensor = torch.tensor(terminated_list, dtype=torch.long).unsqueeze(dim=-1)  # type: ignore[assignment]

        while obs.ndimension() < 2:
            obs = obs.unsqueeze(dim=-1)  # type: ignore[assignment]

        while next_obs.ndimension() < 2:
            next_obs = next_obs.unsqueeze(dim=-1)  # type: ignore[assignment]

        return obs, actions, rewards, next_obs, terminated
