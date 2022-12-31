import sys
sys.path.append(".")
import monazco
from rldog.agents.DQN_based.DQN import DQN
from rldog.configs.FrozenLake_config import FrozenLakeConfig
from rldog.networks.standard_nn import StandardNN
import time
from rldog.tools.logger import logger

if __name__ == '__main__':
    conf = FrozenLakeConfig(is_slippery=False)
    conf.env = monazco.MonazcoEnv()
    conf.n_obs = 14
    conf.n_actions = 14
    net = StandardNN(input_size=14, output_size=14, hidden_size=64, hidden_layers=1)
    conf.DQN_config(network=net, one_hot_encode=False, games_to_play=50000, epsilon_grace_period=0.25, mini_batch_size=8, obs_normalization_factor=18, alpha=0.1, lr=1e-3)
    agent = DQN(conf)  # type: ignore[arg-type]

    start_time = time.time()
    agent.play_games(verbose=False)
    logger.info(f"Training time: {time.time() - start_time:.2f}s")
    agent.evaluate_games(250, plot=False)