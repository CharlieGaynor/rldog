import sys

# sys.path.append('...')
# sys.path.append('..')
sys.path.append(".")
import torch
from rldog.agents.DQN_based.DQN import DQN
from rldog.configs.FrozenLake_config import FrozenLakeConfig
from rldog.dataclasses.DQN_dataclasses import DQN_config
from rldog.tools.logger import logger
import logging

if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    conf = FrozenLakeConfig(is_slippery=False)
    conf.DQN_config()
    agent = DQN(conf)  # type: ignore[arg-type]
    agent.device = torch.device("cpu")
    agent.play_games(2000, verbose=True)
    agent.evaluate_games(100)
