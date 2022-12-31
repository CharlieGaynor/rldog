import sys

sys.path.append(".")
import logging
import time

from rldog.agents.DQN_based.DQN import DQN
from rldog.configs.FrozenLake_config import FrozenLakeConfig
from rldog.networks.networks import StandardNN
from rldog.tools.logger import logger

if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    conf = FrozenLakeConfig(is_slippery=True)

    # Config for just taking the float and using a hidden representation (pick this or the other one)
    # Turns out this sucks and is nearly impossible to win with ( Idk how)
    # net = StandardNN(input_size=1, output_size=4, hidden_size=32, hidden_layers=1)
    # conf.DQN_config(
    #     network=net,
    #     one_hot_encode=False,
    #     games_to_play=5000,
    #     alpha=0.1,
    #     lr=1e-3,
    #     gamma=0.99,
    #     min_epsilon=0.2,
    #     mini_batch_size=2,
    #     epsilon_grace_period=0.25,
    #     obs_normalization_factor=16
    # )
    # agent = DQN(conf, force_cpu=True)  # type: ignore[arg-type]

    # Config for using a one hot encoding representation
    net = StandardNN(input_size=16, output_size=4, hidden_size=16, hidden_layers=1)
    conf.DQN_config(network=net, one_hot_encode=True, games_to_play=5000, epsilon_grace_period=0.25, mini_batch_size=2)
    agent = DQN(conf)  # type: ignore[arg-type]

    start_time = time.time()
    agent.play_games(verbose=False)
    logger.info(f"Training time: {time.time() - start_time:.2f}s")
    agent.evaluate_games(250, plot=False)
