import math

from rewards import BinaryReward
from rewards.base import RewardConfig
from typing import List
from rewards.base import RewardFunction


class ExponentialDecayReward(RewardFunction):
    """
    Exponential decay reward function.
    """

    _tau: float = 1.00
    _base: float = math.e

    @classmethod
    def preprocess(cls, config: RewardConfig) -> None:
        BinaryReward.preprocess(config)

    @classmethod
    def calc_reward(cls, completions: List[str], **kwargs) -> List[float]:
        pattern = BinaryReward._pattern
        rewards: List[float] = []
        for completion in completions:
            matches = pattern.findall(completion)
            forget_count = len(matches)
            reward = cls._base**(-(forget_count / cls._tau)) if cls._base > 0 else 0.0
            rewards.append(reward)
        return rewards