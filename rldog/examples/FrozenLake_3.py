import sys

sys.path.append(".")
import logging
import time

from rldog.agents.actor_critics.A2C import A2C
from rldog.configs.FrozenLake_config import FrozenLakeConfig
from rldog.networks.networks import StandardSoftmaxNN, StandardNN
from rldog.tools.logger import logger

if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    conf = FrozenLakeConfig(is_slippery=False)

    # Config for using a float representation (was too hard for DQN)
    # actor = StandardSoftmaxNN(input_size=1, output_size=4, hidden_size=64, hidden_layers=2)
    # critic = StandardNN(input_size=1, output_size=1, hidden_size=64, hidden_layers=2)
    # conf.actor_critic_config(actor=actor, critic=critic, one_hot_encode=False, games_to_play=6000, lr=2e-3, obs_normalization_factor=16)

    # Config for using a one hot encoding representation
    actor = StandardSoftmaxNN(input_size=16, output_size=4, hidden_size=32, hidden_layers=1)
    critic = StandardNN(input_size=16, output_size=1, hidden_size=32, hidden_layers=1)
    conf.actor_critic_config(
        actor=actor, critic=critic, one_hot_encode=True, games_to_play=500, actor_lr=1e-2, critic_lr=1e-2
    )

    agent = A2C(conf)  # type: ignore[arg-type]

    start_time = time.time()
    agent.play_games(verbose=True)
    logger.info(f"Training time: {time.time() - start_time:.2f}s")
    agent.evaluate_games(100, plot=False)
