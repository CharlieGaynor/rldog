import sys

sys.path.append(".")
import logging
import time

from rldog.agents.policy_gradients.reinforce import Reinforce
from rldog.configs.FrozenLake_config import FrozenLakeConfig
from rldog.networks.networks import StandardSoftmaxNN
from rldog.tools.logger import logger

if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    conf = FrozenLakeConfig(is_slippery=False)

    # Config for using a float representation (was too hard for DQN)
    net = StandardSoftmaxNN(input_size=1, output_size=4, hidden_size=128, hidden_layers=3)
    conf.reinforce_config(network=net, one_hot_encode=False, games_to_play=4000, lr=5e-4, obs_normalization_factor=8)
    
    # Config for using a one hot encoding representation
    # net = StandardSoftmaxNN(input_size=16, output_size=4, hidden_size=32, hidden_layers=1)
    # conf.reinforce_config(network=net, one_hot_encode=True, games_to_play=2000, lr=1e-2)
    
    agent = Reinforce(conf)  # type: ignore[arg-type]

    start_time = time.time()
    agent.play_games(verbose=True)
    logger.info(f"Training time: {time.time() - start_time:.2f}s")
    agent.evaluate_games(100, plot=False)
