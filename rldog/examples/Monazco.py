import sys

sys.path.append(".")
import time

import monazco

from rldog.agents.policy_gradients.reinforce import Reinforce
from rldog.configs.FrozenLake_config import FrozenLakeConfig
from rldog.networks.networks import StandardSoftmaxNN
from rldog.tools.logger import logger

if __name__ == "__main__":

    conf = FrozenLakeConfig(is_slippery=False)
    conf.env = monazco.MonazcoEnv()
    conf.n_obs = 14
    conf.n_actions = 14
    net = StandardSoftmaxNN(input_size=14, output_size=14, hidden_size=64, hidden_layers=2)
    conf.reinforce_config(network=net, one_hot_encode=False, games_to_play=1000, lr=1e-1)
    agent = Reinforce(conf)  # type: ignore[arg-type]

    start_time = time.time()
    agent.play_games(verbose=False)
    logger.info(f"Training time: {time.time() - start_time:.2f}s")
    agent.evaluate_games(250, plot=False)
