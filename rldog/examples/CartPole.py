import sys

sys.path.append(".")
import logging

from rldog.agents.policy_gradients.PPO import PPO
from rldog.configs.CartPole_config import CartPoleConfig
from rldog.tools.logger import logger
import os

if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    conf = CartPoleConfig()
    conf.PPO_config()
    agent = PPO(conf)
    # model_dir = os.path.join(os.getcwd(), 'saved_models/CartPole_PPO_4_2_2_64_03012023')
    # agent.play_games(plot=True, save_model_at_checkpoints=True, model_directory=model_dir)
    # agent.evaluate_games(100, plot=True, log_things=True)
