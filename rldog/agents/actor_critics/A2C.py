from typing import Any, Dict, List, Tuple, Union
import torch
from torch import nn

from rldog.dataclasses.actor_critic_dataclasses import ActorCriticConfig, Transition
from rldog.agents.base_agent import BaseAgent
import random
from rldog.tools.logger import logger
from rldog.tools.plotters import plot_ac_results


class A2C(BaseAgent, ActorCriticConfig):
    """
    Advantage Actor Critic. This is just one worker for now. The actor & critic don't share networks as
    having seperate networks gave better results. The literature is also unclear on which is better.
    """

    def __init__(self, config: ActorCriticConfig, force_cpu: bool = False) -> None:

        self.__dict__.update(config.__dict__)
        super().__init__()

        if force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.transitions: List[Transition] = []
        self.actor_loss: List[float] = []
        self.critic_loss: List[float] = []

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
                logger.info(
                    f"\
Played {game_number} games. Average of last {game_frac} games: reward = {mean(self.reward_averages[-game_frac: ]):.3f}, \
actor loss = {mean(self.actor_loss[-game_frac: ]):.3f}, critic_loss = {mean(self.critic_loss[-game_frac: ]):.3f}"
                )
            while self._network_needs_updating():
                self._update_network()
        if verbose:
            total_rewards = self.reward_averages
            plot_ac_results(total_rewards, actor_loss=self.actor_loss, critic_loss=self.critic_loss)

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
            critic_value = self._get_critic_value(obs)
            next_obs_unformatted, reward, terminated, truncated, info = self.env.step(action)
            next_obs, legal_moves = self._format_obs(next_obs_unformatted, info)
            rewards.append(reward)
            self.transitions.append(Transition(action_probs, reward, critic_value))
            terminated = terminated or truncated

        self.reward_averages.append(sum(rewards))

    def _get_action(  # type: ignore[override]
        self, state: torch.Tensor, legal_moves: List[int] | range, evaluate: bool = False
    ) -> Union[int, Tuple[int, torch.Tensor]]:
        """Sample actions with softmax probabilities. If evaluating, set a min probability"""

        if evaluate:
            with torch.no_grad():
                probabilities: torch.FloatTensor = self.actor(state.to(self.device))
        else:
            probabilities: torch.FloatTensor = self.actor(state.to(self.device))  # type: ignore[no-redef]

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

    def _get_critic_value(self, state: torch.Tensor) -> torch.FloatTensor:
        return self.critic(state)  # type: ignore[no-any-return]

    def _network_needs_updating(self) -> bool:
        return len(self.transitions) > 0

    def _update_network(self) -> None:
        """Sample experiences, compute & back propagate loss"""

        attributes = self._attributes_from_transitions(self.transitions)

        actor_loss, critic_loss = self._compute_losses(*attributes)

        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_value)  # type: ignore[attr-defined]
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_value)  # type: ignore[attr-defined]

        self.actor_opt.step()
        self.critic_opt.step()

        self.transitions = []
        self.actor_loss.append(actor_loss.item())
        self.critic_loss.append(critic_loss.item())

    def _compute_losses(
        self, action_probs: torch.FloatTensor, rewards: List[float], critic_values: torch.FloatTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        discounted_rewards = self._compute_discounted_rewards(rewards)

        # First compute actor loss and update network
        actor_loss = self._compute_actor_loss(action_probs, discounted_rewards, critic_values)
        critic_loss = self._compute_critic_loss(discounted_rewards, critic_values)

        return actor_loss, critic_loss

    def _compute_actor_loss(
        self,
        action_probs: torch.FloatTensor,
        discounted_rewards: torch.FloatTensor,
        critic_values: torch.FloatTensor,
    ) -> torch.Tensor:
        """Compute loss according to REINFORCE"""

        # Really important to detach critic_values, otherwise we are changing the critic network twice and it ends up a mess
        loss = torch.sum(-1.0 * torch.log(action_probs) * (discounted_rewards - critic_values.detach()))
        logger.debug(f"actor loss {loss}")
        return loss

    def _compute_critic_loss(
        self, discounted_rewards: torch.FloatTensor, critic_values: torch.FloatTensor
    ) -> torch.Tensor:
        """Simply the difference between critic value and discounted rewards"""
        # if sum(discounted_rewards) > 1e-3:
        #     logger.debug(f'rewards, critic values: {discounted_rewards}, {critic_values}')
        logger.debug(f"critic loss {torch.sum((discounted_rewards - critic_values) ** 2)}")
        return torch.sum((discounted_rewards - critic_values) ** 2)

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
    ) -> Tuple[torch.FloatTensor, List[float], torch.FloatTensor]:
        """
        Extracts, transforms (and loads, hehe) the attributes hidden in within transitions
        Each resulting tensor should have shape [batch_size, attribute size]
        """
        actions_list = [transition.action_probs for transition in transitions]
        rewards_list = [transition.reward for transition in transitions]
        critic_value_list = [transition.critic_value for transition in transitions]

        # Stack because actions_list contains a list of 0 dimensional tensors
        # action_probs has dimensions [num_actions]
        action_probs: torch.FloatTensor = torch.stack(actions_list, dim=0)  # type: ignore[assignment]
        # Cat because I want a tensor of ndimensions == 1S
        critic_values: torch.FloatTensor = torch.cat(critic_value_list)  # type: ignore

        # logger.debug(f'critical values shape {critic_values.shape}')

        return action_probs, rewards_list, critic_values
