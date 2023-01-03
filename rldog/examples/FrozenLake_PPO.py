import sys

sys.path.append(".")
import logging
import time

from rldog.agents.policy_gradients.PPO import PPO
from rldog.configs.FrozenLake_config import FrozenLakeConfig
from rldog.networks.networks import StandardPPO
from rldog.tools.logger import logger

if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    net = StandardPPO(input_size=16, output_size=4, hidden_layers=2, hidden_size=64)
    old_net = StandardPPO(input_size=16, output_size=4, hidden_layers=2, hidden_size=64)
    conf = FrozenLakeConfig(is_slippery=True)
    conf.PPO_config(
        net=net,
        old_net=old_net,
        one_hot_encode=True,
        games_to_play=75000,
        lr=0.0003,
        n_games_per_learning_batch=2048,
        n_learning_episodes_per_batch=10,
        use_parallel=False,
        n_envs=4,
    )
    agent = PPO(conf)
    start_time = time.time()
    agent.play_games(plot=True)
    logger.info(f"Training time: {time.time() - start_time:.2f}s")
    # agent.evaluate_games(100, plot=False)
