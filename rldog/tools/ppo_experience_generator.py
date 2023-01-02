import random
import torch
import sys
from contextlib import closing

from typing import List, Union, Dict, Tuple, Any
from torch.multiprocessing import Pool
from torch import nn
from random import randint
from rldog.dataclasses.policy_dataclasses import PPOConfig, PPOTransition
from rldog.networks.networks import BasePPONN
import gym


class ParallelExperienceGenerator:
    """Plays n episode in parallel using a fixed agent. Only works for softmax-style policy gradients (i.e. doesn't work for my DQN epsilon greedy)"""

    def __init__(
        self,
        n_actions: int,
        n_obs: int,
        n_episodes: int,
        one_hot_encode: bool,
        obs_norm_factor: float,
        net: BasePPONN,
        env: gym.Env,
        device: torch.device,
        use_parallel: bool = False,
    ):

        self.n_actions = n_actions
        self.n_obs = n_obs
        self.one_hot_encode = one_hot_encode
        self.obs_norm_factor = obs_norm_factor
        self.net = net
        self.device = device
        self.net.to(device)
        self.env = env
        self.use_parallel = use_parallel

        if use_parallel:
            self.n_episodes = max(n_episodes, 61)
            self.net.share_memory()
        else:
            self.n_episodes = n_episodes

    def play_n_episodes(self) -> Tuple[List[List[torch.Tensor]], List[List[int]], List[List[float]]]:
        """Plays n episodes in or not in parallel using the fixed policy and returns the data"""
        if self.use_parallel:
            with closing(Pool(processes=self.n_episodes)) as pool:
                episodes = pool.map(self, range(self.n_episodes))
                pool.terminate()
            return episodes
        else:
            all_states: List[List[torch.Tensor]] = []
            all_actions: List[List[int]] = []
            all_rewards: List[List[float]] = []
            for _ in range(self.n_episodes):
                states, actions, rewards = self._play_1_episode()
                all_states.append(states)
                all_actions.append(actions)
                all_rewards.append(rewards)
            return all_states, all_actions, all_rewards

    def __call__(self, n: int) -> List[PPOTransition]:
        return self._play_1_episode()

    def _play_1_episode(self) -> Tuple[List[torch.Tensor], List[int], List[float]]:
        """Plays 1 episode using the fixed policy and returns the data"""
        next_obs_unformatted, info = self._reset_game()
        next_obs, legal_moves = self._format_obs(next_obs_unformatted, info)
        terminated = False
        states: List[torch.Tensor] = []
        actions: List[int] = []
        rewards: List[float] = []
        while not terminated:
            obs = next_obs
            action = self._get_action(obs, legal_moves)  # type: ignore[misc]
            next_obs_unformatted, reward, terminated, truncated, info = self.env.step(action)
            next_obs, legal_moves = self._format_obs(next_obs_unformatted, info)

            states.append(obs.to(self.device))
            actions.append(action)
            rewards.append(reward)
            terminated = terminated or truncated
        return states, actions, rewards

    def _reset_game(self) -> Tuple[Any, Dict[Any, Any]]:
        """Resets the game environment so it is ready to play a new episode"""
        seed = randint(0, sys.maxsize)
        torch.manual_seed(seed)  # Need to do this otherwise each worker generates same experience
        state, info = self.env.reset()
        return state, info

    def _get_action(self, state: torch.Tensor, legal_moves: List[int] | range) -> int:
        """Sample actions with softmax probabilities. If evaluating, set a min probability"""

        with torch.no_grad():
            probabilities: torch.FloatTensor = self.net.forward_policy(state.to(self.device))

        probs = probabilities.tolist()
        if len(legal_moves) < len(probabilities):
            legal_probs = [probs[i] for i in legal_moves]
            action = random.choices(legal_moves, weights=legal_probs, k=1)[0]
        else:
            action = random.choices(range(len(probs)), weights=probs, k=1)[0]

        return action

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
            new_obs = new_obs / self.obs_norm_factor
            return new_obs, legal_moves
