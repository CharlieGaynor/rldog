from abc import ABC, abstractmethod


class BaseConfig(ABC):
    @abstractmethod
    def __init__(self, max_games: int) -> None:
        pass

    @abstractmethod
    def DQN_config(self) -> None:
        pass
